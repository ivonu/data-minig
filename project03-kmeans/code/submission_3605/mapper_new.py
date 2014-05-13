#!/usr/bin/env python2.7

import sys

if __name__ == "__main__":

    for line in sys.stdin:
        line = line.strip()
        print '1\t%s' % line