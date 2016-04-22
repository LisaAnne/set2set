#!/bin/bash

if [ ! -d prototxts ]; then
  mkdir -p prototxts;
fi;

if [ ! -d results ]; then
  mkdir -p results;
fi;

if [ ! -d snapshots ]; then
  mkdir -p snapshots;
fi;

if [ ! -d results/generated_sentences ]; then
  mkdir -p results/generated_sentences;
fi;

if [ ! -d results/image_features ]; then
  mkdir -p results/image_features;
fi;

