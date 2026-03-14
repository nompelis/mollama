#!/bin/tcsh

# Check that an argument was provided
if ($#argv < 1) then
    echo "Usage: delete_session.csh number"
    exit 1
endif

set num = $argv[1]

#echo "You passed the number: $num"

 curl -vvv http://localhost:11434/api/session/$num -X DELETE

