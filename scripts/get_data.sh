#!/bin/sh
wget 'https://www.uniprot.org/uniprot/?query=annotation:(type:signal)&fil=organism%3A"Homo+sapiens+(Human)+[9606]"+AND+reviewed%3Ayes&format=fasta&force=true&sort=score&compress=yes' \
  -O signal_prots.fasta.gz
wget 'https://www.uniprot.org/uniprot/?query=reviewed%3Ayes+AND+organism%3A"Homo+sapiens+(Human)+[9606]"&format=fasta&force=true&sort=score&compress=yes' \
  -O all_prots.fasta.gz
