#!/bin/bash
set -e

if [ "$1" = 'bash' ]; then
  cd mvpose/backend/light_head_rcnn/lib/ && bash make.sh
fi

cd /mvpose

exec "$@"
