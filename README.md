# pizza-cutter-metadetect

[![tests](https://github.com/beckermr/pizza-cutter-metadetect/actions/workflows/tests.yml/badge.svg)](https://github.com/beckermr/pizza-cutter-metadetect/actions/workflows/tests.yml)

code to run metadetect on pizza cutter MEDs files

## Running `metadetect` on a "pizza slice" MEDS File

The module `pizza_cutter.metadetect` has code to run `metadetect`
on "pizza slice" MEDS files. You can use the command line tool
`run-metadetect-on-slices` like this

```bash
 run-metadetect-on-slices \
     --config=<metadetect config> \
     --seed=<seed> \
     MEDS_FILE_1 MEDS_FILE_2
```
