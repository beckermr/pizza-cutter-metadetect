# pizza-cutter-metadetect

[![tests](https://github.com/beckermr/pizza-cutter-metadetect/actions/workflows/tests.yml/badge.svg)](https://github.com/beckermr/pizza-cutter-metadetect/actions/workflows/tests.yml)

code to run metadetect on pizza cutter MEDs files

## Running the tests w/ actual data

To run the test suite with actual data, do the following:

1. Clone this repo: https://github.com/beckermr/des-y3-test-data
2. Make an environment variable `TEST_DESDATA` that points to where the repo was cloned.

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
