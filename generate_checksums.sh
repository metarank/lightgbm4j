#!/bin/sh

echo "removing all checksums"
find . -name "*.md5" -exec rm {} \;

for f in `find src/main/resources/ -type f`; do
  cat $f | md5sum | awk '{ printf $1 }'> $f.md5
  echo "processed $f"
done