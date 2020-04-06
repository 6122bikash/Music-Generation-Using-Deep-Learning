#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pretty_midi
import os
import numpy as np


# In[1]:


def get_instruments_list(directories):
    list_instrument=[]
    for directory in directories:
        songs=[directory+'/'+filename for filename in os.listdir(directory)]
        for song in songs:
            try:
                midi_data=pretty_midi.PrettyMIDI(song)
            except:
                pass
            for instrument in midi_data.instruments:
                if instrument not in list_instrument:
                    list_instrument.append(instrument)
    return list_instrument


# In[3]:


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


# In[ ]:


def samples_to_midi(samples,program=0,fs=50):
    #reference:https://github.com/craffel/pretty-midi/blob/master/examples/reverse_pianoroll.py
    piano_roll=samples.reshape(-1,96).T
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
    return pm

