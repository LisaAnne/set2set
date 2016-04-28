#!/bin/bash

if [ ! -d prototxts ]; then
  mkdir -p prototxts;
fi;

if [ ! -d snapshots ]; then
  mkdir -p snapshots;
fi;

if [ ! -d logs ]; then
  mkdir -p logs;
fi;

if [ ! -d utils/data ]; then
  mkdir -p utils/data;
fi;

