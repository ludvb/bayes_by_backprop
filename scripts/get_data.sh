#!/bin/sh
wget https://www.uniprot.org/uniprot/?query=reviewed:yes%20annotation:(type:signal)&format=fasta&force=true&sort=score&compress=yes \
  -o signal_prots.fasta.gz
wget https://www.uniprot.org/uniprot/?query=reviewed:yes&format=fasta&force=true&sort=score&compress=yes \
  -o all_prots.fasta.gz
