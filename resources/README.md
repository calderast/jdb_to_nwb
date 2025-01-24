## Channel map
The `channel_map` csv has 256 entries, mapping each electrode to the intan channel it corresponds to.
The channel map for the probe electrodes depends on how the rat was plugged in.  

The `chip_first` column is the channel map for when the chip side (top in eagle) is plugged into first port (preferred). This is the assumed default unless otherwise specified. The `cable_first` column is the channel map for if the cable side (bottom in eagle) was plugged into first port. The cable_first map is simply the chip_first map shifted by ±128, and vice versa (to go from one to the other, subtract 128 from values greater than 128, and add 128 to values less than or equal to 128).

### TODO: Figure out ECOG channels??
Some info I found on ECOG channels so far:

From `plot_impedance_geometry_v3_table`: "If ECoG channel 1+4 connected and plug sequence was "chip first", then channels 128 and 193 (one based) will be low impedance. For "cable first" it is 65 and 256."
```
ecogs.cable=[64 0 191 255]+1; %cable first ecog intan channels / screw 1-4
ecogs.chip=[192 128 63 127]+1; %chip first ecog intan channels
```
They are maybe already handled by being filtered out based on impedance? Check on that.

## Electrode coordinates

The electrode coordinate files contain the relative electrode locations for each probe.
The 3mm and 6mm probes both have 32 shanks, 8 electrodes per shank, and 30um electrode pitch (30um vertically between electrodes on each shank). Odd shanks are vertically offset by 1/2 the electrode pitch (15um).
The 3mm probe has 66um shank pitch (66um horizontally between shanks).
The 6mm probe has 80um shank pitch (80um horizontally between shanks).


The electrode coordinate files originally had 264 entries each: 256 entries for the 256 electrodes + an extra 8 channels recorded by OpenEphys (ADC1 - ADC8). ADC1 records port entry times for alignment, ADC2-ADC8 are unused.
The x, y coordinates for these channels are assigned arbitrarily (they do not actually exist on the probes). They were likely included for alignment purposes in an earlier processing pipeline. I'm not sure if we need them for anything right now, so I have removed them from the csv files but pasted them here for posterity:

Extra 8 channels for 3mm probe:
2376,211
2376,181
2376,151
2376,121
2376,91
2376,61
2376,31
2376,1

Extra 8 channels for 6mm probe:
2880,211
2880,181
2880,151
2880,121
2880,91
2880,61
2880,31
2880,1