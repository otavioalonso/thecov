set -e  # Exit on error

# Detect the operating system (they require slightly different dependencies)
OS=$(uname -s)
ARCH=$(uname -m)

ENVNAME="${1:-thecov}"
echo ENVNAME="$ENVNAME"

echo "Operating system: $OS"
echo "Architecture: $ARCH"

if [[ "$OS" == "Linux" && "$ARCH" == "x86_64" ]]; then
    PLATFORM=linux-64
    export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu/ # <- needed to build mpi4py
elif [[ "$OS" == "Darwin" ]]; then
    PLATFORM=osx-arm64
    echo NOTE: attempting to set environment variables for pyclass on macOS

    # Ensure SDKROOT is set so clang can find system headers
    if command -v xcrun >/dev/null 2>&1; then
        export SDKROOT="$(xcrun --show-sdk-path)"
    fi

    # If Homebrew and libomp are available, configure includes/links
    if command -v brew >/dev/null 2>&1 && brew --prefix libomp >/dev/null 2>&1; then
        LIBOMP_PREFIX="$(brew --prefix libomp)"
        echo "Found libomp at $LIBOMP_PREFIX; configuring include/link flags."
        export CPPFLAGS="-I${LIBOMP_PREFIX}/include ${CPPFLAGS:-}"
        export CFLAGS="-I${LIBOMP_PREFIX}/include ${CFLAGS:-}"
        export CXXFLAGS="-I${LIBOMP_PREFIX}/include ${CXXFLAGS:-}"
        export LDFLAGS="-L${LIBOMP_PREFIX}/lib -Wl,-rpath,${LIBOMP_PREFIX}/lib ${LDFLAGS:-} -lfftw3_threads -lfftw3"
        # -lfftw3_threads -lfftw3"

        # Prefer Homebrew LLVM clang if installed (supports -fopenmp)
        if [ -x "$(brew --prefix llvm)/bin/clang" ]; then
            export CC="$(brew --prefix llvm)/bin/clang"
            export CXX="$(brew --prefix llvm)/bin/clang++"
            echo "Using Homebrew LLVM clang: $CC"
            # allow builds that pass -fopenmp
            export CFLAGS="${CFLAGS} -fopenmp"
            export CXXFLAGS="${CXXFLAGS} -fopenmp"
        else
            # If using Apple clang, the build system may need -Xpreprocessor -fopenmp
            export CC="$(xcrun --find clang)"
            export CXX="$(xcrun --find clang++)"
            echo "Using Apple clang: $CC (will add -Xpreprocessor when needed)"
            export CFLAGS="${CFLAGS} -Xpreprocessor -fopenmp"
            export CXXFLAGS="${CXXFLAGS} -Xpreprocessor -fopenmp"
        fi
    else
        echo "WARNING: libomp not found via brew; install with: brew install libomp."
        echo "WARNING: OpenMP support not available with default MacOS clang, your install may fail when installing pyclass!"
        # fall back to system clang
        export CC="$(xcrun --find clang)"
        export CXX="$(xcrun --find clang++)"
    fi
else
    echo "ERROR: Unsupported platform detected! Installation failed!"
    exit 1
fi

########### conda stuff
# Always execute this script with bash, so that conda shell.hook works.
# Relevant conda bug: https://github.com/conda/conda/issues/7980
if [[ -z "$BASH_VERSION" ]];
then
    exec bash "$0" "$@"
fi

eval "$(conda shell.bash hook)"

echo "Creating new anaconda environment..."
conda create -n "$ENVNAME" python=3.11 "numpy<2.0" -y --platform $PLATFORM
conda activate "$ENVNAME"

conda install -c conda-forge -y openmpi mpi4py matplotlib fftw jupyter

echo "Done! installing thecov..."
python -m pip install -e .

echo Done! activate the new enviornment with: "conda activate '$ENVNAME'"