For recordings with YT bonsai script this is what the .bin AI files are.

Both files are continuous analog-input recordings, sampled at 30 kHz and converted to 32-bit floats. 
They’re written as column-major matrices into the experiment folder, with timestamp suffixes; 
the workflow configures the filenames as AI_.bin and AI_dev4_.bin. (input channels, writers)

File	
AI (Dev3):	0. Camera trigger copy (Dev3/ai0); 1. microphone (ai3); 2. photodiode (ai4); 3. probe sync (ai2); 4. analog-audio copy (ai5); 5. WAV-output copy (ai6).
AI_dev4:	0. Nose-poke beam break (Dev4/ai4); 1. lickport piezo (ai5); 2. pump TTL copy (ai20); 3. probe-sync copy (ai21); 4. red-laser copy (ai19); 5. blue-laser copy (ai18); 6. side photodiode (ai0); 7. airpuff TTL copy (ai22). (Dev4 channel setup)

Camera trigger: generated on Dev3/port0/line0 (40 Hz in the workflow’s default setting); its analog copy is AI channel 0. Camera frames are saved separately as video, not in these .bin files. (trigger and camera setup, video writer)
Ephys/probe sync: generated as a 1 Hz analog signal on Dev3/ao0; recorded as AI channel 3 and as AI_dev4 channel 3. These are the workflow’s probe-sync copies, not the SpikeGLX recording data itself. (sync output)
Audio: the analog output channels are Dev4/ao1 (AnalogAudio) and Dev4/ao3 (AnalogAudioToIR). The workflow records their copies in AI channels 4 and 5; AI_dev4 has no audio channels. These are sampled signal copies, not a standalone WAV file. (audio outputs)