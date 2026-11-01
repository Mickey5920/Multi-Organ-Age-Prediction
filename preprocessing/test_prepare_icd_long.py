"""Invented fixtures for Array pairing and the build/read-only CLI boundary.

Run: python -B preprocessing/test_prepare_icd_long.py --temp-root /project/work/tests
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import prepare_icd_long as icd

TEST_ROOT = None


class ArrayPairingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic-icd-", dir=TEST_ROOT)
        self.root = Path(self.temp.name).resolve()
        assert self.root.is_relative_to(TEST_ROOT)

    def tearDown(self):
        assert self.root.is_relative_to(TEST_ROOT)
        self.temp.cleanup()

    def csv(self, name, headers, rows):
        path = self.root / name
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            writer.writerows(rows)
        return path

    def cli(self, mode, codes, dates, *extra):
        out, error = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(error):
            status = icd.main([mode, "--codes-csv", str(codes), "--dates-csv", str(dates),
                               "--progress-every", "0", *map(str, extra)])
        messages = out.getvalue().splitlines() + error.getvalue().splitlines()
        return status, json.loads(messages[-1])

    def indexed(self):
        # Physical columns and participant rows are independently reordered.
        codes = self.csv("codes.csv", [icd.ID, icd.CODE+" | Array 2", icd.CODE+" | Array 0"],
                         [["SYNTHETIC_A", "E11 Toy second", "I10 Toy first"], ["SYNTHETIC_B", "", ""]])
        dates = self.csv("dates.csv", [icd.ID, icd.DATE+" | Array 0", icd.DATE+" | Array 2"],
                         [["SYNTHETIC_B", "", ""], ["SYNTHETIC_A", "2000/1/2", "2002-03-04"]])
        return codes, dates

    def test_index_keys_and_participant_keys_determine_pairing(self):
        codes, dates = self.indexed()
        output = self.root/"new.csv"
        status, result = self.cli("build", codes, dates, "--output", output)
        self.assertEqual(status, 0, result)
        with output.open(newline="") as stream:
            actual = list(csv.reader(stream))
        self.assertEqual(actual, [icd.OUTPUT,
            ["SYNTHETIC_A", "I10", "Toy first", "2000-01-02", "0"],
            ["SYNTHETIC_A", "E11", "Toy second", "2002-03-04", "2"],
            ["SYNTHETIC_B", "", "", "", ""]])
        before = [p.read_bytes() for p in [codes, dates, output]]
        status, result = self.cli("verify", codes, dates, "--existing-long", output)
        self.assertEqual(status, 0, result)
        self.assertEqual(result["existing"]["mismatched_participants"], 0)
        self.assertEqual(before, [p.read_bytes() for p in [codes, dates, output]])

    def test_middle_missing_date_does_not_shift_later_diagnosis(self):
        rows = icd.pair_arrays({0:"I10 First", 1:"E11 Middle", 2:"J45 Last"},
                              {0:"2000-01-01", 2:"2002-01-01"}, {0,1,2})
        self.assertEqual(rows, [(0,"I10","First","2000-01-01"),
                                (1,"E11","Middle",""), (2,"J45","Last","2002-01-01")])

    def test_packed_middle_empty_position_is_preserved(self):
        codes=self.csv("codes.csv",[icd.ID,icd.CODE],[["SYNTHETIC_A","I10 First||J45 Last"]])
        dates=self.csv("dates.csv",[icd.ID,*[icd.DATE+f" | Array {i}" for i in range(3)]],
                       [["SYNTHETIC_A","2000-01-01","","2002-01-01"]])
        output=self.root/"new.csv"
        status,result=self.cli("build",codes,dates,"--code-format","packed","--packed-order-is-array","--output",output)
        self.assertEqual(status,0,result)
        with output.open(newline="") as stream:r=list(csv.DictReader(stream))
        self.assertEqual([(x["ICD"],x["record_time"],x["source_array_index"]) for x in r],
                         [("I10","2000-01-01","0"),("J45","2002-01-01","2")])

    def test_packed_order_must_be_explicitly_declared(self):
        codes=self.csv("codes.csv",[icd.ID,icd.CODE],[["SYNTHETIC_A","I10 Toy"]])
        dates=self.csv("dates.csv",[icd.ID,icd.DATE+" | Array 0"],[["SYNTHETIC_A","2000-01-01"]])
        output=self.root/"new.csv"
        status,result=self.cli("build",codes,dates,"--code-format","packed","--output",output)
        self.assertEqual(status,2,result)
        self.assertFalse(output.exists())

    def test_orphan_date_is_rejected_and_no_final_file_is_published(self):
        codes,dates=self.indexed()
        dates=self.csv("orphan.csv",[icd.ID,icd.DATE+" | Array 0",icd.DATE+" | Array 2"],
                       [["SYNTHETIC_A","2000-01-02","2002-03-04"],["SYNTHETIC_B","2001-01-01",""]])
        output=self.root/"new.csv"
        status,result=self.cli("build",codes,dates,"--output",output)
        self.assertEqual(status,2,result)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob(".icd-array-*.partial")))

    def test_duplicate_participant_in_either_source_is_rejected(self):
        for duplicate in ["codes","dates"]:
            with self.subTest(duplicate=duplicate):
                codes,dates=self.indexed()
                path=codes if duplicate=="codes" else dates
                with path.open(newline="") as stream:r=list(csv.reader(stream))
                path=self.csv(duplicate+"-duplicate.csv",r[0],r[1:]+[r[1]])
                status,result=self.cli("build",path if duplicate=="codes" else codes,
                                       path if duplicate=="dates" else dates,"--output",self.root/(duplicate+"-out.csv"))
                self.assertEqual(status,2,result)

    def test_missing_participant_key_in_either_source_is_rejected(self):
        for missing in ["codes","dates"]:
            with self.subTest(missing=missing):
                codes,dates=self.indexed()
                path=codes if missing=="codes" else dates
                with path.open(newline="") as stream:r=list(csv.reader(stream))
                path=self.csv(missing+"-missing.csv",r[0],r[1:2])
                status,result=self.cli("build",path if missing=="codes" else codes,
                                       path if missing=="dates" else dates,"--output",self.root/(missing+"-out.csv"))
                self.assertEqual(status,2,result)

    def test_duplicate_array_keys_and_schema_mismatch_are_rejected(self):
        with self.assertRaises(icd.InputError):
            icd.array_columns([icd.CODE+" | Array 0",icd.CODE+" | Array 00"],icd.CODE)
        codes,dates=self.indexed()
        dates=self.csv("mismatch.csv",[icd.ID,icd.DATE+" | Array 0"],[["SYNTHETIC_A","2000-01-01"],["SYNTHETIC_B",""]])
        status,result=self.cli("build",codes,dates,"--output",self.root/"new.csv")
        self.assertEqual(status,2,result)

    def test_invalid_dates_and_nonbaseline_instances_are_rejected(self):
        for value in ["2001-02-29","01/02/2000","2000-13-01"]:
            with self.subTest(value=value),self.assertRaises(icd.InputError):icd.normalize_date(value)
        with self.assertRaises(icd.InputError):
            icd.array_columns([icd.DATE+" | Instance 1 | Array 0"],icd.DATE)

    def test_existing_destination_and_input_are_not_overwritten(self):
        codes,dates=self.indexed();output=self.root/"existing.csv";output.write_text("KEEP ORIGINAL")
        for destination in [output,codes,dates]:
            before=destination.read_bytes()
            status,result=self.cli("build",codes,dates,"--output",destination)
            self.assertEqual(status,2,result)
            self.assertEqual(destination.read_bytes(),before)

    def test_verify_detects_changed_date_and_retains_duplicate_multiplicity(self):
        codes,dates=self.indexed()
        output=self.csv("wrong.csv",icd.OUTPUT[:4],[["SYNTHETIC_A","I10","Toy first","2000-01-02"],
                       ["SYNTHETIC_A","E11","Toy second","2003-03-04"],["SYNTHETIC_B","","",""]])
        status,result=self.cli("verify",codes,dates,"--existing-long",output)
        self.assertEqual(status,1,result)
        self.assertEqual(result["existing"]["mismatched_participants"],1)
        output=self.csv("duplicate.csv",icd.OUTPUT[:4],[["SYNTHETIC_A","I10","Toy first","2000-01-02"],
                       ["SYNTHETIC_A","I10","Toy first","2000-01-02"],
                       ["SYNTHETIC_A","E11","Toy second","2002-03-04"],["SYNTHETIC_B","","",""]])
        status,result=self.cli("verify",codes,dates,"--existing-long",output)
        self.assertEqual(status,1,result)
        self.assertEqual(result["existing"]["mismatched_participants"],1)

    def test_frozen_identity_gate_rejects_wrong_input(self):
        codes,dates=self.indexed()
        output=self.csv("existing.csv",icd.OUTPUT[:4],[])
        before=output.read_bytes()
        status,result=self.cli("verify",codes,dates,"--existing-long",output,"--expected-existing-sha256","0"*64)
        self.assertEqual(status,2,result)
        self.assertEqual(output.read_bytes(),before)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temp-root",type=Path,required=True)
    args,remaining=parser.parse_known_args()
    TEST_ROOT=args.temp_root.resolve()
    TEST_ROOT.mkdir(parents=True,exist_ok=True)
    unittest.main(argv=[sys.argv[0],*remaining],verbosity=2)
