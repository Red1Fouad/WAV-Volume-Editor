import os
import wave
import numpy as np
import struct
import glob

def read_wav_chunks(file):
    """Reads all the chunks in a WAV file, including the 'smpl' chunk."""
    with open(file, 'rb') as f:
        data = f.read()
    
    riff = data[:4].decode('ascii')
    if riff != 'RIFF':
        raise ValueError("Not a valid RIFF file")

    file_size = struct.unpack('<I', data[4:8])[0]
    wave_header = data[8:12].decode('ascii')
    if wave_header != 'WAVE':
        raise ValueError("Not a valid WAV file")
    
    pos = 12
    chunks = {}
    
    while pos + 8 <= len(data):  # need at least 8 bytes for header
        chunk_id = data[pos:pos+4].decode('ascii', errors="replace")
        chunk_size = struct.unpack('<I', data[pos+4:pos+8])[0]
        
        start = pos + 8
        end = start + chunk_size
        if end > len(data):
            break  # avoid reading past EOF
        
        chunk_data = data[start:end]
        chunks[chunk_id] = chunk_data
        
        pos = end
        if chunk_size % 2 == 1:  # padding byte
            pos += 1
    
    return chunks, file_size

def write_wav_chunks(chunks, output_file, original_size):
    """Writes chunks back into a new WAV file, preserving the 'smpl' chunk."""
    with open(output_file, 'wb') as f:
        # Write the RIFF header
        f.write(b'RIFF')
        f.write(struct.pack('<I', original_size))
        f.write(b'WAVE')
        
        # Write all the chunks
        for chunk_id, chunk_data in chunks.items():
            f.write(chunk_id.encode('ascii'))
            f.write(struct.pack('<I', len(chunk_data)))
            f.write(chunk_data)

def reduce_volume(input_wav, output_wav, db_change):
    # Read WAV file chunks, including the smpl chunk
    chunks, original_size = read_wav_chunks(input_wav)
    
    # Extract the audio data chunk ('data')
    if 'data' not in chunks:
        raise ValueError("'data' chunk not found in the WAV file")
    
    audio_frames = chunks['data']
    
    # Read the audio file metadata using the wave module
    with wave.open(input_wav, 'rb') as wav_in:
        params = wav_in.getparams()
        n_channels = wav_in.getnchannels()
        sampwidth = wav_in.getsampwidth()
        framerate = wav_in.getframerate()
        n_frames = wav_in.getnframes()
    
    # Print file metadata
    print(f"Processing file: {input_wav}")
    print(f"Channels: {n_channels}")
    print(f"Sample width: {sampwidth} bytes")
    print(f"Frame rate (Sample rate): {framerate} Hz")
    print(f"Number of frames: {n_frames}")
    
    # Convert the byte data to numpy array for processing
    if sampwidth == 1:  # 8-bit audio (unsigned)
        dtype = np.uint8
        audio_data = np.frombuffer(audio_frames, dtype=dtype).astype(np.float32)
        audio_data -= 128  # Convert to signed
    elif sampwidth == 2:  # 16-bit audio (signed)
        dtype = np.int16
        audio_data = np.frombuffer(audio_frames, dtype=dtype).astype(np.float32)
    elif sampwidth == 4:  # 32-bit audio (signed)
        dtype = np.int32
        audio_data = np.frombuffer(audio_frames, dtype=dtype).astype(np.float32)
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")
    
    # Calculate the factor for volume change (positive for reduce, negative for amplify)
    volume_change_factor = 10 ** (db_change / 20)
    
    # Apply volume change
    audio_data *= volume_change_factor
    
    # Clamp values to prevent clipping
    max_val = 2 ** (8 * sampwidth - 1) - 1
    min_val = -max_val - 1
    audio_data = np.clip(audio_data, min_val, max_val)
    
    # Convert back to the original integer type
    audio_data = audio_data.astype(dtype)
    
    # Convert back to original byte format
    if sampwidth == 1:
        audio_data += 128  # Convert back to unsigned 8-bit
    new_frames = audio_data.tobytes()
    
    # Replace the 'data' chunk with the modified frames
    chunks['data'] = new_frames
    
    # Calculate new file size (RIFF size is the total size minus 8 bytes)
    new_size = original_size - len(chunks['data']) + len(new_frames)
    
    # Write the new WAV file, preserving the original structure
    write_wav_chunks(chunks, output_wav, new_size)

def process_wav_files_in_directory():
    # Ask the user whether to reduce or amplify the volume (r for reduce, a for amplify)
    action = input("Do you want to reduce (r) or amplify (a) the volume? (r/a): ").strip().lower()
    
    if action not in ["r", "a"]:
        print("Invalid choice. Please choose 'r' for reduce or 'a' for amplify.")
        return
    
    # Ask the user for the dB change amount
    try:
        db_amount = float(input("Enter the dB amount: ").strip())
    except ValueError:
        print("Invalid dB amount. Please enter a valid number.")
        return

    # If amplifying, the dB amount should be negative (to increase volume)
    if action == "a":
        db_amount = -db_amount  # Negative value to amplify the volume
    
    # Get the current directory (where the script is located)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create an output directory if it doesn't exist
    output_dir = os.path.join(current_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .wav files in the current directory
    wav_files = glob.glob(os.path.join(current_dir, '*.wav'))
    
    if not wav_files:
        print("No .wav files found in the current directory.")
        return
    
    # Process each .wav file
    for wav_file in wav_files:
        # Construct output file path (same name but in the output folder)
        output_wav = os.path.join(output_dir, os.path.basename(wav_file))
        
        # Reduce or amplify the volume of the WAV file
        reduce_volume(wav_file, output_wav, db_change=db_amount)
        print(f"Output file created: {output_wav}")

# Run the processing function
process_wav_files_in_directory()

