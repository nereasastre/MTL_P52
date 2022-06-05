from mido import MidiFile

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
OCTAVES = list(range(7))
NOTES_IN_OCTAVE = len(NOTES)
errors = {
    'program': 'Bad input, please refer this spec-\n'
               'http://www.electronics.dit.ie/staff/tscarff/Music_technology/midi/program_change.htm',
    'notes': 'Bad input! note must be in [0, 127]"-\n'
}


def read_midi(file_name):
    """
    Reads the given midi_file
    Args:
        file_name (str): The name/path of the file to be read as MIDI

    Returns:
        The MIDI file read with mido
    """

    return MidiFile(file_name, clip=True)


def list_number_to_list_note(midi_notes, note_off=False):
    """
    Converts a list of note events coming from the midi_info function to a list of tuples with (note from A-G, octave)
    Args:
        midi_notes (list(list)): A list of note events coming from the midi_info function where each note event has
        format [note_on/note_off, midi_note, time, channel]
        note_off (bool): If True, takes into account the note_off events. Otherwise it just converts to note the note_on
        events.

    Returns:
        A list of tuples with the note from A-G and the octave of each note event in the midi_notes list

    """
    def _number_to_note(number: int) -> tuple:
        """
        Aux private function to convert a single midi number to A-G note and octave
        Args:
            number: the MIDI number to convert to note (A-G)

        Returns:
            A tuple containing the note (A-G) and octave
            TODO do we want this as a single string? I feel like having it as a separate thing might make more sense
        """
        # nerea: added a -1 because we were an octave off, TODO check this does not give us problems in the future
        octave = number // NOTES_IN_OCTAVE - 1
        assert octave in OCTAVES, errors['notes']
        assert 0 <= number <= 127, errors['notes']
        note = NOTES[number % NOTES_IN_OCTAVE]

        return note, octave

    notes_list = []
    for note_event in midi_notes:
        if note_event[0] == 'time_signature':  # skip first sublist since it corresponds to the time signature
            continue
        else:
            if not note_off and note_event[0] == 'note_off':  # skip note_off if note_off disabled
                continue
            notes_list.append(_number_to_note(note_event[1])) # convert note and append it to notes_list
    return notes_list


def midi_info(file_name):
    """
    Reads midi file and returns a list of the note events.
    Args:
        file_name: The file name/path of the midi file to be read

    Returns:
        A list of list. Each sublist list represents a note event and has format
        [type, note, time, channel], where type represents note_on/note_off, note is the note (0-127) in midi format,
        time is the starting time of the event and channel is the channel in which the event happens.

    """
    mid = read_midi(file_name)
    midi_dict = []
    output = []

    # Put all note on/off in midi note as dictionary.
    for i in mid:
        if i.type == 'note_on' or i.type == 'time_signature':
            midi_dict.append(i.dict())

    mem1 = 0
    for i in midi_dict:
        # Change time values to accumulated time
        time = i['time'] + mem1
        i['time'] = time
        mem1 = i['time']

        # Make every note_on with 0 velocity note_off (velocity 0 represents the end of a note)
        if i['type'] == 'note_on' and i['velocity'] == 0:
            i['type'] = 'note_off'

        # Put note, start_time, stop_time, as nested list in a list. Format is [type, note, time, channel]
        # TODO do we want the channel????
        mem2 = []
        if i['type'] == 'note_on' or i['type'] == 'note_off':
            mem2.append(i['type'])
            mem2.append(i['note'])
            mem2.append(i['time'])
            mem2.append(i['channel'])
            output.append(mem2)

        # Put time signatures (tipus de compas)
        if i['type'] == 'time_signature':
            mem2.append(i['type'])
            mem2.append(i['numerator'])
            mem2.append(i['denominator'])
            mem2.append(i['time'])
            output.append(mem2)

    return output


output = midi_info('twinkle_twinkle.mid')
print(output)
print(list_number_to_list_note(output))
