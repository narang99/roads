#!/bin/bash

# manage subdirectories as tar files and dvc
# you need to remember which ones you tarred with
# faster pull and push on cloud (dvc is slow for numerous files)
# tar is slower though
# TODO: do this if you need to add this to colab


add() {
    # do tar compress, then run dvc add
    direc="$1"
    parent_dir="$(dirname $direc)"
    dirn="$(basename $direc)"
    (cd $parent_dir && tar -cvf "$dirn.tar" "$dirn")
    dvc add "$dirn.tar"
    echo "Add the directory $direc in .gitignore too please"
}

pull() {
    direc="$1"
    parent_dir="$(dirname $direc)"
    dirn="$(basename $direc)"

    dvc pull "$parent_dir/$dirn.tar"
    (cd "$parent_dir" && tar -xvf "$dirn")
}
# Command parsing
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 {add|pull} <directory>"
    echo "  add <directory>   - Compress directory to tar and add to dvc"
    echo "  pull <directory>  - Pull tar from dvc and extract directory"
    exit 1
fi

command="$1"
shift

case "$command" in
    add)
        add "$@"
        ;;
    pull)
        pull "$@"
        ;;
    *)
        echo "Unknown command: $command"
        echo "Use 'add' or 'pull'"
        exit 1
        ;;
esac