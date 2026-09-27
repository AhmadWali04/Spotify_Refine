"""PRD risk: test the Essentia and CLAP installs early (during M1), before M4 needs them.

    python check_audio_install.py
"""
for name, mod in [("Essentia (TensorFlow build)", "essentia.standard"), ("LAION-CLAP", "laion_clap")]:
    try:
        __import__(mod)
        print(f"OK       {name}")
    except Exception as e:  # noqa: BLE001
        print(f"MISSING  {name}: {type(e).__name__}: {e}")
