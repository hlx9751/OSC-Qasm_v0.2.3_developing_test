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
_IBM_RUNTIME_AVAILABLE = True

Sampler = None
import argparse
import sys
import eel
import socket
import asyncio
import numpy as np
import importlib
import re
import time
import os
import subprocess
import datetime

provider = None
CLOUD_TOKEN = None

LOG_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'osc_qasm_debug.log')

def log_to_file(msg):
    try:
        with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
            ts = datetime.datetime.now().strftime('%H:%M:%S')
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass

CLOUD_INSTANCE = None
CLOUD_REGION = None
CLOUD_ERR = None

# Normalize instance input to avoid runtime crashes when CRN is malformed
CNR_PATTERN = re.compile(r"^crn:v\d:bluemix:public:quantum-computing:[\w-]+:a/[^:]+:[^:]+::?$")

def _normalize_instance(inst):
    if not inst:
        return "open-instance"
    s = str(inst).strip()
    if s.lower() == "open-instance":
        return "open-instance"
    if inst.startswith("crn:"):
        return inst
    if CNR_PATTERN.match(s):
        return s
    return "open-instance"

def _normalize_region(r):
    if not r:
        return "us-east"
    s = str(r).strip().lower()
    if s in ("us-east", "eu-de"):
        return s
    return "us-east"

def _canonical_backend_name(name):
    s = str(name).strip()
    low = s.lower()
    # Accept snake_case fake backend names from Max/Pd and map to qiskit class naming.
    # Example: fake_torino -> FakeTorino
    if low.startswith('fake_'):
        tail = low[5:]
        return 'Fake' + ''.join(part.capitalize() for part in tail.split('_') if part)
    return s

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
        ss = str(s)
        counts[ss] = counts.get(ss, 0) + 1
    return counts

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
        p01 = None
        p10 = None
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

def _apply_readout_noise_from_backend(counts, shots, backend):
    props = None
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
        sbs = str(bs)
        out[sbs] = out.get(sbs, 0) + 1
    return out

def _run_local_job(qc, shots, backend):
    tc = transpile(qc, backend)
    return backend.run(tc, shots=shots)

def _configure_runtime(token, region, instance):
    global provider, CLOUD_TOKEN, CLOUD_INSTANCE, CLOUD_REGION
    provider = None
    CLOUD_TOKEN = token if token and token != "false" else None
    CLOUD_INSTANCE = instance if instance else None
    CLOUD_REGION = region if region else 'us-east'
    if CLOUD_TOKEN:
        ensure_provider()

def _resolve_local_ip(remote_value):
    if remote_value in (None, "None"):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    if remote_value not in (False, "false"):
        return remote_value
    return "127.0.0.1"

def ensure_provider():
    global provider
    token = CLOUD_TOKEN
    instance = _normalize_instance(CLOUD_INSTANCE)
    region = _normalize_region(CLOUD_REGION)
    if provider:
        return provider
    try:
        if not (_IBM_RUNTIME_AVAILABLE and token):
            provider = None
            return provider
        try:
            provider = QiskitRuntimeService(
                channel="ibm_cloud",
                token=token,
                instance=instance or "open-instance",
                region=region,
            )
        except TypeError:
            try:
                provider = QiskitRuntimeService(
                    channel="ibm_cloud",
                    cloud_api_key=token,
                    instance=instance or "open-instance",
                    cloud_region=region,
                )
            except Exception:
                provider = None
        except Exception:
            provider = None
    except Exception as e:
        uiprint(str(e))
        provider = None
    return provider

def _direct_cloud_sampler(backend_name, shots, qc):
    try:
        global CLOUD_ERR
        CLOUD_ERR = None

        srv = ensure_provider()

        if not srv:
            CLOUD_ERR = "Could not initialize QiskitRuntimeService. Check API Key/CRN."
            return None

        uiprint(f"Qiskit Runtime: getting backend {backend_name}...")
        backend = srv.backend(backend_name)
        uiprint(f"Qiskit Runtime: transpiling for {backend_name}...")
        isa_qc = transpile(qc, backend=backend, optimization_level=1)
        uiprint("Qiskit Runtime: submitting job...")

        if SamplerV2:
            sampler = SamplerV2(mode=backend)
            job = sampler.run([isa_qc], shots=int(shots))
        elif Sampler:
            sampler = Sampler(backend=backend)
            job = sampler.run(circuits=[isa_qc], shots=int(shots))
        else:
            CLOUD_ERR = "Qiskit Runtime Sampler is unavailable."
            return None

        uiprint(f"Job ID: {job.job_id()}")
        deadline = time.time() + 180
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
                CLOUD_ERR = f"Runtime job failed: {st_s}"
                return None

            time.sleep(2)

        if not (last_status and any(x in last_status for x in done_states)):
            CLOUD_ERR = f"Runtime timeout waiting for job {job.job_id()} (last status: {last_status or 'unknown'})"
            return None

        result = job.result()
        if SamplerV2:
            pub_result = result[0]
            data = pub_result.data
            for field_name in data:
                field_val = getattr(data, field_name)
                if hasattr(field_val, 'get_counts'):
                    return field_val.get_counts()
            CLOUD_ERR = "Runtime result did not contain counts."
            return None

        return result.quasi_dists[0].binary_probabilities()
    except Exception as e:
        CLOUD_ERR = f"Runtime Error: {e}"
        return None

class FileLikeOutputOSC(object):
    ''' This class emulates a File-Like object
        with a "write()" method that can be used
        by print() and qiskit.tools.job_monitor()
        as an alternative output (replacing sys.stdout)
        to send messages through the OSC-Qasm client

        usage: print("foo", file=FileLikeOutputOSC())
        '''
    def __init__(self):
        pass

    def write(self, text):
        if text != f'\n' and text != "": # Skips end='\n'|'' argument messages
            print(text) # uiprint back to console
            
            # Log to file
            log_to_file(f"[STDOUT] {text}")

            # Send message body back to Max on info channel
            # We strip the timestamp if present to keep it clean
            msg = text
            if len(msg) > 12 and msg[12] == ':': # naive timestamp check
                 msg = msg[12:]
            
            # Send to Max
            client.send_message("/info", msg)
            
            # ALSO SEND TO TERMINAL (Standard Output)
            # This ensures you can see it in the VS Code / System terminal
            # independent of the GUI log
            sys.__stdout__.write(f"[LOG] {text}\n")

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

            # Log to file
            log_to_file(f"[STDERR] {text}")

            # Log to terminal for debugging
            sys.__stdout__.write(f"[ERR] {text}\n")

            if text == ERR_SEP and self.older != ERR_SEP and self.older != "": # There is a line like ERR_SEP both at the begining and end of a qiskit error log!
                # uiprint the last entry before the ending ERR_SEP
                client.send_message("/error", "error in OSC-Qasm server: \n(...) "+self.older+"switch to console to learn more")

            elif "KeyboardInterrupt" in text:
                # When closing the program with Ctrl+c, There is a 'KeyboardInterrupt' error message.
                client.send_message("/info", "OSC-Qasm Server has Stopped.")

            self.older=text # Update memory


def run_circuit(qc, shots, backend_name):
    backend_name = _canonical_backend_name(backend_name)
    if backend_name == 'ibmq_qasm_simulator':
        backend_name = 'qasm_simulator'
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
            cls = getattr(fake_mod, backend_name)
            backend = cls()
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

    if CLOUD_TOKEN:
        uiprint("Attempting Qiskit Runtime...")
        counts = _direct_cloud_sampler(backend_name, shots, qc)
        if counts is not None:
            return counts
        if CLOUD_ERR:
            uiprint(f"Qiskit Runtime failed: {CLOUD_ERR}")
            try:
                client.send_message("/error", CLOUD_ERR)
            except Exception:
                pass
        return {}

    msg = f"Backend '{backend_name}' requires IBM credentials."
    uiprint(msg)
    client.send_message("/error", msg)
    return {}


def parse_qasm(*args):
    global qc
    qc=QuantumCircuit().from_qasm_str(args[1])
    if len(args)>2:
        try:
            shots = int(args[2])
        except Exception:
            shots = 1024
        pass
    else:
        shots=1024

    if len(args)>3:
        backend_name = args[3]
    else:
        backend_name='qasm_simulator'

    counts = run_circuit(qc, shots, backend_name)
    uiprint("Sending result counts back to Client")
    client.send_message("/info", "Retrieving results from OSC-Qasm..." )
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

# Mapping the OSC Server callback function
callback = dispatcher.Dispatcher()
callback.map("/QuTune", parse_qasm)


def CLI(UDP_IP, RECEIVE_PORT, SEND_PORT, TOKEN, HUB, PROJECT, REMOTE):

    global client, ERR_SEP
    ERR_SEP = '----------------------------------------' # For FileLikeErrorOSC() class
    _configure_runtime(TOKEN, HUB, PROJECT)
    if UDP_IP=="localhost":
        UDP_IP="127.0.0.1"

    local_ip = _resolve_local_ip(REMOTE)

    #OSC server and client
    server = osc_server.ThreadingOSCUDPServer((local_ip, RECEIVE_PORT), callback)
    client = udp_client.SimpleUDPClient(UDP_IP, SEND_PORT)
    client.send_message("/info", "OSC-Qasm is now running")
    uiprint("Server Receiving on {} port {}".format(server.server_address[0], server.server_address[1]))
    uiprint("Server Sending back on {} port {}".format(client._address,  client._port))
    server.serve_forever()


async def server_process(args):
    global client, ERR_SEP
    ERR_SEP = '----------------------------------------' # For FileLikeErrorOSC() class

    #OSC server and client
    #parsing arguments from GUI
    wUDP_IP = args[0]
    wRECEIVE_PORT = int(args[1])
    wSEND_PORT = int(args[2])
    wTOKEN = args[3]
    wHUB = args[4]
    wPROJECT = args[5]
    wREMOTE = args[6]
    _configure_runtime(wTOKEN, wHUB, wPROJECT)
    if wUDP_IP=="localhost":
        wUDP_IP="127.0.0.1"
    local_ip = _resolve_local_ip(wREMOTE)
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


if __name__ == '__main__':
    global HEADLESS
    p = argparse.ArgumentParser()

    p.add_argument('receive_port', type=int, nargs='?', default=1416, help='The port where the OSC-Qasm Server will listen for incoming messages. Default port is 1416')
    p.add_argument('send_port', type=int, nargs='?', default=1417, help='The port that OSC-Qasm will use to send messages back to the Client (the client\'s listening port). Default port is 1417')
    p.add_argument('ip', nargs='?', default='127.0.0.1', help='The IP address to where the retrieved results will be sent to (Where the Client is located). Default IP is 127.0.0.1 (localhost)')
    p.add_argument('--token', help='IBM Cloud API Key / Runtime token for running circuits on IBM hardware')
    p.add_argument('--hub', default='us-east', help='IBM Runtime region, for example us-east or eu-de')
    p.add_argument('--project', default='open-instance', help='IBM Runtime instance or CRN, defaults to open-instance')
    p.add_argument('--remote', nargs='?', default=False, help='Declare this as a remote server. In this case, OSC-Qasm will be listenning to messages coming into the network adapter address. If there is a specific network adapter IP you want to listen in, add it as an argument here')
    p.add_argument('--headless', nargs='?', type=bool, const=True, default=False, help='Run OSC-Qasm in headless mode. This is useful if you don\'t want to launch the GUI and only work in the terminal.')

    args = p.parse_args()

    # Route sys.stderr to OSC
    flerr = FileLikeErrorOSC()
    sys.stderr = flerr

    HEADLESS = args.headless

    if not HEADLESS:
        eel.init('GUI')

    def uiprint(*message):
        text = " ".join(map(str, message))
        log_to_file(f"[UI] {text}")
        
        if HEADLESS:
            print(*message)
        else:
            eel.print(*message)

    # Initialize log file
    with open(LOG_FILE_PATH, 'w', encoding='utf-8') as f:
        f.write(f"--- OSC-Qasm Log Started at {datetime.datetime.now()} ---\n")

    # Launch separate terminal window to tail the log
    try:
        if not HEADLESS: # Only pop up window if in GUI mode
            uiprint(f"Opening debug log window for: {LOG_FILE_PATH}")
            cmd = f'tail -f "{LOG_FILE_PATH}"'
            subprocess.Popen(['osascript', '-e', f'tell application "Terminal" to do script "{cmd}"'])
    except Exception as e:
        print(f"Failed to open log window: {e}")

    uiprint('================================================')
    uiprint(' OSC_QASM by OCH & Itaborala @ QuTune (v2.1.2) ')
    uiprint(' https://iccmr-quantum.github.io               ')
    uiprint('================================================')

    if HEADLESS:
        CLI(args.ip, args.receive_port, args.send_port, args.token, args.hub, args.project, args.remote)
    else:
        GUI()
