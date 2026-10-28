"""Load an explicitly selected native runtime before optional numeric packages.

Usage: python -B scripts/run_native_entry.py --library LIBRARY entry.py [args...]
The entry retains its own source/input checks and solver lifecycle.
"""
from pathlib import Path
import argparse,os,runpy,sys
from rinkspace.core.native import NativeCore


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--library',required=True,type=Path)
 parser.add_argument('entry',type=Path)
 parser.add_argument('arguments',nargs=argparse.REMAINDER)
 args=parser.parse_args()
 # Keep the native runtime and DLL-directory handles alive for the entry.
 held_core=NativeCore(args.library)
 entry=args.entry.resolve(strict=True)
 sys.path.insert(0,str(entry.parent));sys.argv=[str(entry),*args.arguments]
 runpy.run_path(str(entry),run_name='__main__')
 return held_core


if __name__=='__main__':main()
