#!/bin/bash
# Update system
apt update && apt upgrade -y

apt install -y git wget gcc

# Install conda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

# Initialize conda for both bash and zsh
~/miniconda3/bin/conda init bash && ~/miniconda3/bin/conda init zsh

source ~/.bashrc

# Create environment
conda env create -f .devcontainer/environment.yml
conda activate linc

# Install Prover9
mkdir -p ~/prover9
git clone https://github.com/theoremprover-museum/prover9.git ~/prover9
cd ~/prover9
git fetch origin pull/2/head:pr-2
git checkout 99f6d9011c38ba976a3bcb6d42b3a795908c68c0
cd ~/prover9/source
mkdir -p ~/prover9/source/bin
make all
mkdir -p /usr/local/bin/prover9
cp -r /root/prover9/source/bin/* /usr/local/bin/prover9/
