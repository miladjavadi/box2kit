"""
generate conditionally random midi sequence.

mostly written by chatgpt.
"""

import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage
import random

# Configuration
note_set = [36, 38, 40, 42, 43, 45, 46, 47, 48, 49, 51]
bpm = 90
ticks_per_beat = 480
num_bars = 10000
beats_per_bar = 4
subdivisions = 2 

mid = MidiFile(ticks_per_beat=ticks_per_beat)
track = MidiTrack()
mid.tracks.append(track)

tempo = mido.bpm2tempo(bpm)
track.append(MetaMessage('set_tempo', tempo=tempo))

step = ticks_per_beat // subdivisions  # ticks per 8th note
duration = step // 2  # note length = half an 8th note
total_steps = num_bars * beats_per_bar * subdivisions

# We’ll keep track of delta times explicitly
time_since_last = 0

for step_idx in range(total_steps):
    chosen_notes = random.sample(note_set, k=random.randint(0, 3))
    velocity = random.randint(80, 120)

    # Step delay before the first note_on of the step
    if step_idx == 0:
        delta_time = 0  # no delay before first note ever
    else:
        delta_time = step - duration  # rest of the step after previous note_off

    # First note_on gets the delta time
    if chosen_notes:
        track.append(Message('note_on', note=chosen_notes[0], velocity=velocity, time=delta_time))
        # other note_ons are simultaneous, time=0
        for note in chosen_notes[1:]:
            track.append(Message('note_on', note=note, velocity=velocity, time=0))
    else:
        # no notes this step, just advance time
        # add a dummy wait event with no notes to hold timing
        track.append(Message('note_on', note=0, velocity=0, time=delta_time))
        track.append(Message('note_off', note=0, velocity=0, time=0))
        continue

    # Note offs: first one after duration
    track.append(Message('note_off', note=chosen_notes[0], velocity=velocity, time=duration))
    # others simultaneous, time=0
    for note in chosen_notes[1:]:
        track.append(Message('note_off', note=note, velocity=velocity, time=0))


mid.save('no_gap_random_chords.mid')
