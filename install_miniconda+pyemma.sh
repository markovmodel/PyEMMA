#!/usr/bin/env bash
# This script installs Miniconda and PyEMMA. If conda is already available, we install PyEMMA in a new environment.
# By default, we use the name 'pyemma' as environment name.
# The desired output folders and env names can be set by arguments.
# Author: Martin Scherer, 2018

function usage {
    echo "Usage of installation script:"
    echo "bash $0 -t 'miniconda_target_directory' path to Miniconda"
    echo "bash $0 -e 'conda_environment' environment name in which pyemma will be installed."
    echo "bash $0 -i ignore existing miniconda installation. Get a fresh copy."
    echo "bash $0 -n do not add the new path of miniconda installation to PATH (bashrc update) [off by default]."
    echo "bash $0 -m minimal set of packags, eg. omit jupyter notebook installation."
}

# default values for arguments.
target=$HOME/miniconda3
env_name="pyemma"
ignore=1 # dont ignore existing miniconda installation, but just use it to install our environment.
no_add_bashrc=1 # add new target to PATH in .bashrc
minimal=1 # minimal set of packags, eg. omit jupyter notebook installation


while getopts "ht:e:inm" opt; do
   case $opt in
       n) no_add_bashrc=0;;
       m) minimal=0;;
       t) target=$OPTARG;;
       e) env_name=$OPTARG;;
       i) ignore=0;;
       h) usage
         exit 0;;
       *) echo "unknown option/combination"; exit 0;;
   esac
done


function add_to_bashrc {
    if [ $no_add_bashrc -ne 0 ]; then
        return 0
    fi

	# add path of target to bashrc, if not yet present.
	# note: copied from miniconda installer.
    BASH_RC="$HOME"/.bashrc
    if [ -f "$BASH_RC" ]; then
        printf "\\n"
        printf "Appending source %s/bin/activate to %s\\n" "$target" "$BASH_RC"
        printf "A backup will be made to: %s-miniconda3.bak\\n" "$BASH_RC"
        printf "\\n"
        cp "$BASH_RC" "${BASH_RC}"-miniconda3.bak
    else
        printf "\\n"
        printf "Appending source %s/bin/activate in\\n" "$target"
        printf "newly created %s\\n" "$BASH_RC"
    fi
    printf "\\n"
    printf "For this change to become active, you have to open a new terminal.\\n"
    printf "\\n"
    printf "\\n" >> "$BASH_RC"
    printf "# added by Miniconda3 installer\\n"            >> "$BASH_RC"
    printf "export PATH=\"%s/bin:\$PATH\"\\n" "$target"  >> "$BASH_RC"
}


function install_miniconda {
    echo "installing miniconda to ${target}..."
    if [ `uname -m` = "x86_64"  ]; then
        arch="x86_64"
    else
        arch="x86"
    fi

    platform=`uname -s`
    if [ ${platform} = "Darwin" ]; then
        platform="MacOSX"
        if [ $arch = "x86" ]; then
            echo "ERROR: 32 bit on OSX is not supported by Anaconda, please compile the software yourself."
            exit 1
        fi
    fi

	echo "installing miniconda to $target"
	f=`mktemp`
	# curl should be available on Linux and OSX.
	curl https://repo.continuum.io/miniconda/Miniconda3-latest-$platform-$arch.sh -o ${f}
	bash ${f} -b -f -p ${target}
	export PATH=${target}/bin:$PATH
	hash -r
    add_to_bashrc
}


function install_pyemma {
    # installs pyemma in an env named pyemma
    packages="pyemma numpy"
    if [ ! minimal ]; then
        # add optional packages.
        packages="$packages notebook ipython ipywidgets"
    fi
    # we add numpy here to get the openblas variant (instead of MKL).
    echo "creating pyemma environment..."
    conda=${target}/bin/conda
    if [[ -z $($conda env list | grep ${env_name}) ]]; then
        echo "env '$env_name' does not exist. Create new one."
        $conda create -n ${env_name} ${packages} -y -c conda-forge
	source activate ${env_name}
        $conda config --env --add channels conda-forge
	echo "execute the following command to activate your newly created pyemma environment:"
	echo "> source activate ${env_name}"
    else
        echo "environment with name='${env_name}' already exists. Please specify a different name by -e new_name"
        exit 1
    fi
}

# probe for conda, install miniconda otherwise.
if [ -x "$(command -v conda)" ] && [ $ignore -eq 1 ]; then
    echo "Found conda. Will install to a new environment named ${env_name}"
    install_pyemma
else
    if [ -d ${target} ]; then
        echo "ERROR: Destination directory for Miniconda $target already exists, please choose another one."
        exit 1
    else
        install_miniconda
        install_pyemma
    fi
fi
