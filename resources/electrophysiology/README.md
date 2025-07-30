# Electrophysiology resources

## Probes:
All probes used by the Berke Lab in the hex maze task are listed in `ephys_devices.yaml`. Berke Lab currently uses custom silicon probes designed by Daniel Egert for high-density hippocampal recordings. 

The “first gen” probes have 256 channels (32 shanks x 8 electrodes per shank). There is 30um between electrodes on each shank. Odd shanks are vertically offset by 1/2 the electrode pitch (15um). Shanks are numbered "left to right": 1 (leftmost shank) to 32 (rightmost shank). Electrodes are numbered "top to bottom": 1 (top, most dorsal) to 8 (tip, most ventral) on each shank. These probes come in 3 different lengths: 3mm probe (with 66um shank pitch), 6mm probe (with 80um shank pitch), and 9mm probe (with 100um shank pitch). The 9mm probe has never been used in the hex maze, so is not included in `ephys_devices.yaml`.

The “next gen” probes have 252 channels (21 shanks x 12 electrodes per shank). There is 25um between electrodes on each shank. Odd shanks are vertically offset by 1/2 the electrode pitch (12.5um). Shanks are numbered "left to right": 1 (leftmost shank) to 21 (rightmost shank). Electrodes are numbered "bottom to top": 1 (tip, most ventral) to 12 (top, most dorsal) on each shank. These probes come in 2 different lengths: 4mm probe (with 80um shank pitch) and 10mm probe (with 100um shank pitch).

Berke Lab has also done some pilot recordings with Neuropixels 2.0 (4-shank) probes. Full support for Neuropixels will be added if we decide to move forwards with Neuropixels recordings in the hex maze.

## Electrode coordinates:
Electrode coordinate files (and corresponding figures) for each probe contain the relative x,y locations of each electrode on the probe. These files and figures were generated based on the known probe geometry by `generate_electrode_coordinates.ipynb`.

## Channel map:
The Intan boards we use have 256 channels (2x 128 channel headstage). The channel map is needed to map which Intan channel corresponds to each recording electrode on our probe.

The channel map depends on how the rat was plugged in (which of the 2 SPI cables is plugged into which port). The “chip first” map is the channel map for when the chip side (top in Eagle) is plugged into the first port. This is preferred and is the assumed default unless otherwise specified. The “cable first” map is the channel map for if the cable side (bottom in Eagle) was plugged into the first port. The “cable first” map is simply the “chip first” map shifted by ±128, and vice versa (to go from one to the other, subtract 128 from values greater than 128, and add 128 to values less than or equal to 128).

The 256-channel probes and the 252-channel probes follow the same routing of “electrode number” to Intan channel. Electrode number is what we would get if we assigned an absolute number to each electrode on the probe. For 256-channel probes (8 electrodes per shank), shank 1 has electrode numbers 1-8, shank 2 has electrode numbers 9-16, etc. For 252-channel probes (12 electrodes per shank), shank 1 has electrode numbers 1-12, shank 2 has electrode numbers 13-24, etc.

*There are a few key things to note:*
- The same electrode number is not in the same place on the 256-channel probe and the 252-channel probe. For example, electrode 23 is S03E07 on the 256-channel probe and S02E11 on the 252-channel probe.
- On the 256-channel probes, electrodes are numbered "top to bottom": 1 (top, most dorsal) to 8 (tip, most ventral) on each shank. On the 252-channel probes, electrodes are numbered "bottom to top": 1 (tip, most ventral) to 12 (top, most dorsal) on each shank. Shanks are labeled “left to right” on both probes. For example, S01E01 on a 256-channel probe can be considered the “top left” electrode, but S01E01 on a 252-channel probe is the “bottom left” electrode.
- The “electrode number” to Intan channel correspondence between the 256-channel probes and 252-channel probes is slightly offset by the 4 missing electrodes in the 252-channel probes. These channels can be used for ECoG screws (see note below). The channel maps for the 256-channel probes and 252-channel probes would be identical if we placed the ECoG channel locations where the missing electrodes are. Instead we list these 4 channels at the end of the channel map for clarity.


## Note on ECoG channels:

For the 256-channel probes, we may connect up to 4 ECOG screws. If these screws are connected, they replace the following electrodes and are recorded by the following Intan channels:

- S03E08/SCREW2 - Intan channel 128 (chip_first) or 0 (cable_first)
- S09E01/SCREW1 - Intan channel 192 (chip_first) or 64 (cable_first)
- S24E08/SCREW4 - Intan channel 127 (chip_first) or 255 (cable_first)
- S30E01/SCREW3 - Intan channel 63 (chip_first) or 191 (cable_first)

We can tell which channels (if any) are connected to ECoG screws because they will be low impedance.

For the 252-channel probes, 4 of the Intan channels are not connected to electrodes. These can be used for ECoG screws:

- SCREW3: between S02E11 and S02E12 - Intan channel 128 (chip_first) or 0 (cable_first)
- SCREW4: between S06E03 and S06E04 - Intan channel 192 (chip_first) or 64 (cable_first)
- SCREW1: between S16E09 and S16E10 - Intan channel 127 (chip_first) or 255 (cable_first)
- SCREW2: between S20E01 and S20E02 - Intan channel 63 (chip_first) or 191 (cable_first)