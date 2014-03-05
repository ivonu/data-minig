#!/usr/bin/env python2
import sys
import re

# DEBUG: read from file in argument instead of stdin
if len(sys.argv) > 1:
    sys.stdin = open(sys.argv[1], 'r')

p = re.compile('[a-z]+')
#--- get all lines from stdin ---
for line in sys.stdin:
    #--- remove leading and trailing whitespace---
    line = line.strip()
    #--- split the line into words ---
    words = line.split()
    #--- output tuples [word, 1] in tab-delimited format---
    for word in words:
        word = word.lower()
        m = p.match(word)
        if m:
            print '%s\t%s' % (m.group(0), "1")
