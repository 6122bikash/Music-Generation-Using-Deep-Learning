#import modules
import pretty_midi
import os
import numpy as np
import pickle


def get_instruments_list(directories):
    list_instrument=[]
    for directory in directories:
        songs=[directory+'/'+filename for filename in os.listdir(directory)]
        for song in songs:
            try:
                midi_data=pretty_midi.PrettyMIDI(song)
            
                for instrument in midi_data.instruments:
                    if instrument not in list_instrument:
                        list_instrument.append(instrument)
            except:
                pass
    return list_instrument


def get_notes(directories='dataset',sampling_frequency=50,save_at='notes/session_note'):
    notes=[]
    if(os.path.exists(save_at)):
        with open(save_at,'rb') as f:
            notes=pickle.load(f)
    else:
        for directory in directories:
            print(directory)
            songs=[directory+'/'+filename for filename in os.listdir(directory)]
            #Building piano_roll
            for song in songs:
                try:
                    midi_data=pretty_midi.PrettyMIDI(song)
                    piano_roll=midi_data.get_piano_roll(fs=sampling_frequency,times=None)
                    for i in range(piano_roll.shape[1]):
                        indexes=np.where(piano_roll[:,i]>0)[0]
                        a=[str(c) for c in indexes]
                        note=','.join(a)
                        notes.append(note)
                except:
                    pass
    n_vocab=len(set([item for item in notes]))
    n_notes=len(notes)
    with open(save_at,'wb') as f:
        pickle.dump(notes,f)
    print("number of samples:"+str(n_notes))
    print("vocab size:"+str(n_vocab)) 

    return notes


def prepare_sequences(notes,is_LSTM=False):
    sequence_length = 100

    # get all Bitch names
    pitchnames = sorted(set(item for item in notes))
    n_vocab=len(pitchnames)

    # create a dictionary to map Bitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, int((len(notes) - sequence_length)/sequence_length), 1):
        sequence_in = notes[sequence_length*i:sequence_length+i*sequence_length]
        sequence_out=notes[sequence_length+i*sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(sequence_out)

    # reshape the input into a format compatible with LSTM layers
    network_output=np.asarray(network_output)
    network_input=np.array(network_input)
    network_input=np.eye(n_vocab)[network_input]
    if is_LSTM==True:
        return network_input,network_output
    else:
        network_input2=np.roll(network_input,1,axis=1)
        network_input2[:,0,:]=0
        return (network_input, network_input2)



#IGNORE IGNORE IGNORE this function for now
def midi_datasets_to_samples(directories,sampling_frequency,save_at='sample.npy'):
    samples=[]
    start_pitch_index=16
    end_pitch_index=112
    n_vocab=end_pitch_index-start_pitch_index
    for directory in directories:
        print(directory)
        songs=[directory+'/'+filename for filename in os.listdir(directory)]
        for song in songs:
            print(song)
            try:
                midi_data=pretty_midi.PrettyMIDI(song)
            except:
                pass
            piano_roll=midi_data.get_piano_roll(fs=sampling_frequency,times=None).T
            sample=piano_roll[0:96*int(piano_roll.shape[0]/96),start_pitch_index:end_pitch_index]
            samples=samples+[sample]  #append is computationally faster than concatention so uing list append instead of numpy concatenation
            
    
    samples=np.concatenate(samples,axis=0)
    samples=samples.reshape(samples.shape[0]//96,96,96)
    samples.astype(np.int8)
    np.save(save_at,samples)    
    return samples


def sample_to_piano_roll(notes,decoded):
    pitchname=sorted(set(item for item in notes))
    int_to_note = dict((number, note) for number, note in enumerate(pitchname))
    gen_note=[int_to_note[ix] for ix in decoded]
    piano_roll=[]
    for note in gen_note:
        a=[0]*128
        for index in note.split(','):
            if(index==''):
                index=0
            a[int(index)]=40
        piano_roll.append(a)
        
    piano_roll=np.array(piano_roll).T
    return piano_roll


def piano_roll_to_midi(piano_roll,program=0,fs=50,save_at='sample.midi'):
    #reference:https://github.com/craffel/pretty-midi/blob/master/examples/reverse_pianoroll.py
    notes,frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    pm.write('sample.midi')
    return pm

