#!/bin/tcsh

# Check that an argument was provided
if ($#argv < 1) then
    echo "Usage: chat.csh number"
    exit 1
endif

set num = $argv[1]

 echo "You passed the session number: $num"

 curl -vvv http://localhost:11434/api/chat -X POST \
      -H "Content-Type: application/json" \
      -d '{"session_id": '$num', "message": "Hello shim world."}'

