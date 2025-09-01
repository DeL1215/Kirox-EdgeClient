# list_audio_devices.py
import sounddevice as sd

KEYWORDS = ["headset", "headphone", "usb", "bluetooth", "realtek", "mic", "microphone"]

def normalize(s):
    return (s or "").lower()

def main():
    try:
        devices = sd.query_devices()
    except Exception as e:
        print("Failed to query devices:", e)
        return

    default_dev = sd.default.device
    def_in = getattr(default_dev, "input", default_dev[0] if isinstance(default_dev, (list, tuple)) else None)
    def_out = getattr(default_dev, "output", default_dev[1] if isinstance(default_dev, (list, tuple)) else None)

    print("==== Default devices ====")
    print("Default input :", def_in)
    print("Default output:", def_out)
    print()

    print("==== INPUT devices (for microphone/VAD) ====")
    for i, d in enumerate(devices):
        name = d.get("name", f"dev{i}")
        ins = int(d.get("max_input_channels", 0))
        if ins > 0:
            sr = d.get("default_samplerate", None)
            tag = ""
            lowname = normalize(name)
            if any(k in lowname for k in KEYWORDS):
                tag = "  <-- maybe your headset?"
            print(f"#{i:2d} | inputs={ins:2d} | default_sr={sr} | {name}{tag}")

    print()
    print("==== OUTPUT devices (for playback) ====")
    for i, d in enumerate(devices):
        name = d.get("name", f"dev{i}")
        outs = int(d.get("max_output_channels", 0))
        if outs > 0:
            sr = d.get("default_samplerate", None)
            lowname = normalize(name)
            tag = ""
            if any(k in lowname for k in KEYWORDS):
                tag = "  <-- maybe your headset?"
            print(f"#{i:2d} | outputs={outs:2d} | default_sr={sr} | {name}{tag}")

   

if __name__ == "__main__":
    main()
