# Real Dataset Protocol

No generated modulation corpus is used in the experiment.

## WIDEFT

- Publisher: New Mexico State University.
- DOI: `10.5281/zenodo.4110980`.
- License: CC BY 4.0.
- Archive checksum: `985702f80e97d5631bca519f56a0eec1`.
- Real data: 100 captured bursts for each of 138 devices; Bluetooth, Wi-Fi,
  and other protocols.
- 10% rule: select exactly 10 of 100 bursts for every device/protocol stratum
  by seeded SHA-256 order.
- Acquisition: inspect the remote ZIP directory, then range-download only those
  selected burst entries and verify each entry's CRC32.

## INRIA PLA

- Publisher: Inria Lille and University of Cambridge.
- DOI: `10.5281/zenodo.18268648`.
- License: CC BY 4.0.
- Archive checksum: `aff583bee6f4efccd08fe78c731bf03d`.
- Real data: raw complex64 I/Q captures from 12 physical devices.
- 10% rule: enumerate non-overlapping physical capture windows, then select
  the first aligned 10% of each device capture during streaming extraction.
- Acquisition: stop decompression after 10% of each physical capture; the full
  archive and full device files are never stored in the Studio.
- Audit unit: provenance reports the aligned uncompressed-byte fraction rather
  than claiming that 12 selected device prefixes are 10% of 12 device files.
- Leakage control: selected windows are sorted by physical offset before
  contiguous train/validation/test partitioning.

## RF Fingerprinting WLAN Dataset

- Publisher: Wroclaw University of Science and Technology.
- DOI: `10.5281/zenodo.18515187`.
- License: CC BY 4.0.
- Real data: NPZ captures from physical WLAN transmitters in an anechoic
  chamber and an office.
- 10% rule: select 10% per transmitter/environment stratum by seeded SHA-256
  order.
- Acquisition: inspect both remote ZIP directories, then range-download only
  the selected NPZ entries and verify each entry's CRC32.
- Domain-generalization test: train/validate on anechoic captures and test
  only on office captures.

## Auditability

The generated manifest is the experiment's source of truth. It records every
selected sample and split. It also stores DOI, license, source URL, archive
size, and publisher checksum in the adjacent JSON metadata file.
