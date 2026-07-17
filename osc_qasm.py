# OSC-Qasm
# A simple OSC Python interface for executing Qasm code.
# Or a simple way to connect creative programming environments like Max (The QAC Toolkit) and Pd with real quantum hardware, using the OSC protocol.
#
# Omar Costa Hamido / Paulo Vitor Itaboraí (2021 - 2022)
# https://github.com/iccmr-quantum/OSC-Qasm
#

from pythonosc import dispatcher, osc_server, udp_client
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
import argparse
import sys
import eel
import socket
import asyncio
import numpy as np
import importlib
import time

#✅ Normalize instance input to avoid runtime crashes when Instance (CRN) is malformed
def _normalize_instance(instance):
    if not instance:
        return "open-instance"
    s = str(instance).strip()
    if s.lower() == "open-instance":
        return "open-instance"
    if s.startswith("crn:"):
        return s
    return "open-instance"

#✅ Normalize region input 
def _normalize_region(region):
    if not region:
        return "us-east"
    s = str(region).strip().lower()
    if s in ("us-east", "eu-de"):
        return s
    return "us-east"

#✅ Normalize backend name input
def _normalize_backend_name(backend_name):
    s = str(backend_name).strip()
    low = s.lower()
    # Accept snake_case fake backend names from Max/Pd and map to qiskit class naming.
    # Example: fake_torino -> FakeTorino
    if low.startswith('fake_'):
        tail = low[5:]
        return 'Fake' + ''.join(part.capitalize() for part in tail.split('_') if part)
    return s

#✅ Local statevector simulator 
#❓ This is different from the IBM Runtime, MicroQiskit and och.microqisit approaches, so it may need comparison tests.
def _local_statevector_counts(qc, shots):
    try:
        qc_nm = qc.remove_final_measurements(inplace=False)
    except Exception:
        qc_nm = QuantumCircuit(qc.num_qubits)
        for instr, qargs, cargs in qc.data:
            if instr.name != 'measure':
                qc_nm.append(instr, qargs, cargs)
    state = Statevector.from_int(0, dims=(2,) * qc_nm.num_qubits).evolve(qc_nm)
    probs_dict = state.probabilities_dict()
    keys = list(probs_dict.keys())
    probs = np.array([probs_dict[k] for k in keys], dtype=float)
    if probs.sum() == 0:
        probs = np.ones(len(keys)) / len(keys)
    samples = np.random.choice(keys, size=int(shots), p=probs)
    counts = {}
    for s in samples:
        counts[s] = counts.get(s, 0) + 1
    return counts

#✅ readout error simulation
def _readout_error_probs(props, qubit):
    p01 = None
    p10 = None
    try:
        for item in props.qubits[int(qubit)]:
            if getattr(item, "name", None) == "prob_meas0_prep1":
                p01 = float(item.value)
            elif getattr(item, "name", None) == "prob_meas1_prep0":
                p10 = float(item.value)
    except Exception:
        pass
    if p01 is None or p10 is None:
        try:
            ro = float(props.readout_error(int(qubit)))
            p01 = ro / 2.0
            p10 = ro / 2.0
        except Exception:
            return None, None
    p01 = min(max(p01, 0.0), 1.0)
    p10 = min(max(p10, 0.0), 1.0)
    return p01, p10

#✅ Apply readout noise snapshot
def _apply_readout_noise_from_backend(counts, shots, backend):
    try:
        props = backend.properties()
    except Exception:
        props = None
    if props is None:
        raise RuntimeError("Fake backend does not expose backend.properties(); cannot apply noise snapshot.")

    if not counts:
        return {}

    nbits = max(len(str(k)) for k in counts.keys())
    total = sum(int(v) for v in counts.values()) or int(shots)
    dist = {}
    for k, v in counts.items():
        bs = str(k).zfill(nbits)
        dist[bs] = dist.get(bs, 0.0) + (int(v) / total)

    for q in range(nbits):
        p01, p10 = _readout_error_probs(props, q)
        if p01 is None or p10 is None:
            raise RuntimeError(f"Missing readout error snapshot for qubit {q}.")
        bit_pos = nbits - 1 - q
        new_dist = {}
        for bs, p in dist.items():
            b = bs[bit_pos]
            if b == "0":
                bs0 = bs
                bs1 = bs[:bit_pos] + "1" + bs[bit_pos + 1 :]
                new_dist[bs0] = new_dist.get(bs0, 0.0) + p * (1.0 - p10)
                new_dist[bs1] = new_dist.get(bs1, 0.0) + p * p10
            else:
                bs1 = bs
                bs0 = bs[:bit_pos] + "0" + bs[bit_pos + 1 :]
                new_dist[bs1] = new_dist.get(bs1, 0.0) + p * (1.0 - p01)
                new_dist[bs0] = new_dist.get(bs0, 0.0) + p * p01
        dist = new_dist

    keys = list(dist.keys())
    probs = np.array([dist[k] for k in keys], dtype=float)
    s = probs.sum()
    if s <= 0:
        probs = np.ones(len(keys)) / len(keys)
    else:
        probs = probs / s
    samples = np.random.choice(keys, size=int(shots), p=probs)
    out = {}
    for bs in samples:
        out[bs] = out.get(bs, 0) + 1
    return out

#✅ setup Runtime
def _configure_runtime(token, region, instance):
    runtime = {
        "provider": None,
        "token": token if token and token != "false" else None,
        "instance": _normalize_instance(instance),
        "region": _normalize_region(region),
    }
    if runtime["token"]:
        ensure_provider(runtime)
    return runtime

#✅ Local IP resolution 
#❓ need test
def _resolve_local_ip(remote_value):
    if remote_value in (False, "false"):
        return "127.0.0.1"
    if remote_value not in (None, "None"):
        return remote_value

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    finally:
        sock.close()

#✅ Runtime provider initialization
#❓ This needs testing with a real IBM Runtime login. Compared with the last known working version, I removed one layer of nested try/except.
def ensure_provider(runtime):
    token = runtime["token"]
    instance = runtime["instance"]
    region = runtime["region"]
    if runtime["provider"]:
        return runtime["provider"]
    if not token:
        return None
    try:
        runtime["provider"] = QiskitRuntimeService(
            channel="ibm_cloud",
            token=token,
            instance=instance,
            region=region,
        )
    except TypeError:
        try:
            runtime["provider"] = QiskitRuntimeService(
                channel="ibm_cloud",
                cloud_api_key=token,
                instance=instance,
                cloud_region=region,
            )
        except Exception as e:
            uiprint(str(e))
    except Exception as e:
        uiprint(str(e))
    return runtime["provider"]

#✅ Direct cloud sampler
#❓ Needs testing on a real IBM Quantum backend. Compared with the last known working version, the legacy sampler-result compatibility code has been removed because this path now targets SamplerV2 only.
def _direct_cloud_sampler(runtime, backend_name, shots, qc):
    try:
        srv = ensure_provider(runtime)

        if not srv:
            return None, "Could not initialize QiskitRuntimeService. Check API Key/CRN."

        uiprint(f"Qiskit Runtime: getting backend {backend_name}...")
        backend = srv.backend(backend_name)
        uiprint(f"Qiskit Runtime: transpiling for {backend_name}...")
        isa_qc = transpile(qc, backend=backend, optimization_level=1)
        uiprint("Qiskit Runtime: submitting job...")

        sampler = SamplerV2(mode=backend)
        job = sampler.run([isa_qc], shots=int(shots))

        uiprint(f"Job ID: {job.job_id()}")
        deadline = time.time() + 180 #Adjust the timeout here if needed. A GUI timeout parameter could be added later.
        last_status = None
        done_states = {'done', 'completed'}
        fail_states = {'error', 'failed', 'cancelled', 'canceled'}

        while time.time() < deadline:
            try:
                st_s = str(job.status()).lower()
            except Exception:
                st_s = 'unknown'

            if st_s != last_status:
                uiprint(f"Runtime job status: {st_s}")
                last_status = st_s

            if any(x in st_s for x in done_states):
                break
            if any(x in st_s for x in fail_states):
                return None, f"Runtime job failed: {st_s}"

            time.sleep(2)

        if not (last_status and any(x in last_status for x in done_states)):
            return None, f"Runtime timeout waiting for job {job.job_id()} (last status: {last_status or 'unknown'})"

        result = job.result()
        pub_result = result[0]
        data = pub_result.data
        for field_name in data:
            field_val = getattr(data, field_name)
            if hasattr(field_val, 'get_counts'):
                return field_val.get_counts(), None
        return None, "Runtime result did not contain counts."
    except Exception as e:
        return None, f"Runtime Error: {e}"

#✅ File-like error handler, same as the old version
class FileLikeErrorOSC(object):
    ''' This class emulates a File-Like object
        with a "write()" method that can be used
        to pipe Qiskit error messages through
        the OSC-Qasm client

        usage: sys.stderr = FileLikeErrorOSC()
        '''
    def __init__(self):

        self.older="" # stderr 'memory'

    def write(self, text):
        if text != f'\n' and text != "": # Skips end='\n'|'' argument messages
            print(text) # uiprint back to console

            if text == ERR_SEP and self.older != ERR_SEP and self.older != "": # There is a line like ERR_SEP both at the begining and end of a qiskit error log!
                # uiprint the last entry before the ending ERR_SEP
                client.send_message("/error", "error in OSC-Qasm server: \n(...) "+self.older+"switch to console to learn more")

            elif "KeyboardInterrupt" in text:
                # When closing the program with Ctrl+c, There is a 'KeyboardInterrupt' error message.
                client.send_message("/info", "OSC-Qasm Server has Stopped.")

            self.older=text # Update memory


#✅ Run circuit on selected backend
def run_circuit(runtime, qc, shots, backend_name):
    backend_name = _normalize_backend_name(backend_name)
    is_fake = backend_name.startswith("Fake")

    uiprint("Running circuit on {}...".format(backend_name))
    client.send_message("/info", "Running circuit on {}...".format(backend_name))

    if backend_name == "qasm_simulator":
        counts = _local_statevector_counts(qc, shots)
        uiprint("Done!")
        return counts

    if is_fake:
        try:
            fake_mod = importlib.import_module("qiskit_ibm_runtime.fake_provider")
            backend_cls = getattr(fake_mod, backend_name)
            backend = backend_cls()
        except Exception as e:
            msg = f"Fake backend '{backend_name}' is unavailable: {e}"
            uiprint(msg)
            client.send_message("/error", msg)
            return {}

        try:
            transpile(qc, backend=backend, optimization_level=1)
            base_counts = _local_statevector_counts(qc, shots)
            noisy_counts = _apply_readout_noise_from_backend(base_counts, int(shots), backend)
            uiprint("Done!")
            return noisy_counts
        except Exception as e:
            msg = f"Fake backend '{backend_name}' execution failed: {e}"
            uiprint(msg)
            client.send_message("/error", msg)
            return {}

#✅ Runtime sampler
    if runtime["token"]:
        uiprint("Attempting Qiskit Runtime...")
        counts, cloud_err = _direct_cloud_sampler(runtime, backend_name, shots, qc)
        if counts is not None:
            return counts
        if cloud_err:
            uiprint(f"Qiskit Runtime failed: {cloud_err}")
            try:
                client.send_message("/error", cloud_err)
            except Exception:
                pass
        return {}

    msg = f"Backend '{backend_name}' requires IBM credentials."
    uiprint(msg)
    client.send_message("/error", msg)
    return {}

#✅ Parse QASM input and send counts back over OSC, basically same with the old version, except for the shots argument.
def parse_qasm(runtime, qasm_text, shots_arg=1024, backend_name='qasm_simulator'):
    qc = QuantumCircuit().from_qasm_str(qasm_text)
    if shots_arg is not None:
        try:
            shots = int(shots_arg)
        except Exception:
            shots = 1024
    else:
        shots = 1024

    counts = run_circuit(runtime, qc, shots, backend_name)
    uiprint("Sending result counts back to Client")
    client.send_message("/info", "Retrieving results from OSC-Qasm...")
    # list comprehension that converts a Dict into an
    # interleaved string list: [key1, value1, key2, value2...]
    sorted_counts = {}
    for key in sorted(counts):
        #uiprint ("%s: %s" % (key, counts[key]) )
        sorted_counts[key]=counts[key]
    counts_list = [str(x) for z in zip(sorted_counts.keys(), sorted_counts.values()) for x in z]
    # and then into a string
    counts_list = " ".join(counts_list)
    client.send_message("/counts", counts_list)

#✅ Build a runtime-aware OSC dispatcher for /QuTune. Here includes deglobalize the callback of Qutune
def _build_dispatcher(runtime):
    callback = dispatcher.Dispatcher()

    def _handle_qasm(_address, *osc_args):
        if not osc_args:
            return
        qasm_text = osc_args[0]
        shots = osc_args[1] if len(osc_args) > 1 else 1024
        backend_name = osc_args[2] if len(osc_args) > 2 else 'qasm_simulator'
        parse_qasm(runtime, qasm_text, shots, backend_name)

    callback.map("/QuTune", _handle_qasm)
    return callback

#✅ basically same with the old version, with `local_ip` and `callback` extracted into helper functions
#❓ If GUI works, then 'async def server_process(args)' works, then CLI should work. but still need to try CLI
def CLI(UDP_IP, RECEIVE_PORT, SEND_PORT, TOKEN, REGION, INSTANCE, REMOTE):

    global client, ERR_SEP
    ERR_SEP = '----------------------------------------' # For FileLikeErrorOSC() class
    runtime = _configure_runtime(TOKEN, REGION, INSTANCE)
    if UDP_IP=="localhost":
        UDP_IP="127.0.0.1"

    local_ip = _resolve_local_ip(REMOTE)
    callback = _build_dispatcher(runtime)

    #OSC server and client
    server = osc_server.ThreadingOSCUDPServer((local_ip, RECEIVE_PORT), callback)
    client = udp_client.SimpleUDPClient(UDP_IP, SEND_PORT)
    client.send_message("/info", "OSC-Qasm is now running")
    uiprint("Server Receiving on {} port {}".format(server.server_address[0], server.server_address[1]))
    uiprint("Server Sending back on {} port {}".format(client._address,  client._port))
    server.serve_forever()

#✅ basically same with the old version, `wGroup`/`wHUB` and `wPROJECT` consolidated into `wREGION` and `wINSTANCE`
async def server_process(args):
    global client, ERR_SEP
    ERR_SEP = '----------------------------------------' # For FileLikeErrorOSC() class

    #OSC server and client
    #parsing arguments from GUI
    wUDP_IP = args[0]
    wRECEIVE_PORT = int(args[1])
    wSEND_PORT = int(args[2])
    wTOKEN = args[3]
    wREGION = args[4]
    wINSTANCE = args[5]
    wREMOTE = args[6]
    runtime = _configure_runtime(wTOKEN, wREGION, wINSTANCE)
    if wUDP_IP=="localhost":
        wUDP_IP="127.0.0.1"
    local_ip = _resolve_local_ip(wREMOTE)
    callback = _build_dispatcher(runtime)
    server = osc_server.AsyncIOOSCUDPServer((local_ip, wRECEIVE_PORT), callback, asyncio.get_event_loop())
    transport, _protocol = await server.create_serve_endpoint()
    client = udp_client.SimpleUDPClient(wUDP_IP, wSEND_PORT)
    client.send_message("/info", "OSC-Qasm is now running")
    uiprint("Server Receiving on {} port {}".format(server._server_address[0], server._server_address[1]))
    uiprint("Server Sending back on {} port {}".format(client._address,  client._port))
    while server_on:
        eel.sleep(0.333)
        await asyncio.sleep(0.333)
    transport.close()
    uiprint("Server has stopped now.")
    client.send_message("/info", "OSC-Qasm Server has Stopped.")

#✅ basically same with the old version, deleted pythonPrint(received) 
def GUI():
    @eel.expose
    def start(*args):
        global server_on
        server_on = True
        asyncio.run(server_process(args))
    @eel.expose
    def stop():
        global server_on
        server_on = False
    eel.start('index.html', cmdline_args=['-incognito'],size=(840,480),block=True)

#✅ basically same with the old version, fix and rename '--hub';'--group'; '--project' to '--region'/'--instance', same changes are made in osc_qasm.maxref.xml, GUI/index.html, and README.md
if __name__ == '__main__':
    global HEADLESS
    p = argparse.ArgumentParser()

    p.add_argument('receive_port', type=int, nargs='?', default=1416, help='The port where the OSC-Qasm Server will listen for incoming messages. Default port is 1416')
    p.add_argument('send_port', type=int, nargs='?', default=1417, help='The port that OSC-Qasm will use to send messages back to the Client (the client\'s listening port). Default port is 1417')
    p.add_argument('ip', nargs='?', default='127.0.0.1', help='The IP address to where the retrieved results will be sent to (Where the Client is located). Default IP is 127.0.0.1 (localhost)')
    p.add_argument('--token', help='IBM Cloud API Key / Runtime token for running circuits on IBM hardware')
    p.add_argument('--region', default='us-east', help='IBM Runtime region, for example us-east or eu-de')
    p.add_argument('--instance', default='open-instance', help='IBM Runtime instance or CRN, defaults to open-instance')
    p.add_argument('--remote', nargs='?', default=False, help='Declare this as a remote server. In this case, OSC-Qasm will be listenning to messages coming into the network adapter address. If there is a specific network adapter IP you want to listen in, add it as an argument here')
    p.add_argument('--headless', nargs='?', type=bool, const=True, default=False, help='Run OSC-Qasm in headless mode. This is useful if you don\'t want to launch the GUI and only work in the terminal.')

    args = p.parse_args()

    # Route sys.stderr to OSC
    flerr = FileLikeErrorOSC()
    sys.stderr = flerr


#✅ drop the old args.token / hub / group / project validation path in favor of the new Runtime setup flow
    HEADLESS = args.headless

    if not HEADLESS:
        eel.init('GUI')

    def uiprint(*message):
        if HEADLESS:
            print(*message)
        else:
            eel.print(*message)

#❓ need some changes here?
    uiprint('================================================')
    uiprint(' OSC_QASM by OCH & Itaborala @ QuTune (v2.1.2) ')
    uiprint(' https://iccmr-quantum.github.io               ')
    uiprint('================================================')

    if HEADLESS:
        CLI(args.ip, args.receive_port, args.send_port, args.token, args.region, args.instance, args.remote)
    else:
        GUI()
