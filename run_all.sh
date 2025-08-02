#!/bin/bash
set -e

# 1. Create environment
env_name="crafter_env"
PYTHON_VERSION=3.10

if command -v conda &> /dev/null; then
    echo "[INFO] Conda detected. Creating conda environment: $env_name"
    if ! conda env list | grep -q "$env_name"; then
        conda create -y -n $env_name python=$PYTHON_VERSION
    fi

    # 2. Install dependencies with conda first
    if [ -f requirements.txt ]; then
        echo "[INFO] Installing dependencies from requirements.txt using conda (and pip fallback)"
        while IFS= read -r line; do
            pkg=$(echo "$line" | sed 's/[=<>!].*//')
            [[ "$pkg" =~ ^#.*$ || -z "$pkg" ]] && continue
            if conda search "$pkg" --info > /dev/null 2>&1; then
                echo "[CONDA] Installing $line"
                conda run -n $env_name conda install -y "$line"
            else
                echo "[WARN] $pkg not found in conda, trying pip..."
                conda run -n $env_name pip install "$line"
            fi
        done < requirements.txt
    else
        echo "[ERROR] requirements.txt not found!"
        exit 1
    fi

    # 3. Run main simulation
    if [ -f run.py ]; then
        echo "[INFO] Running run.py simulation..."
        conda run -n $env_name python run.py
    else
        echo "[ERROR] run.py not found!"
        exit 1
    fi

    # 4. Make multi-agent video
    if [ -f make_video_from_results.py ]; then
        echo "[INFO] Generating multi-agent panel video..."
        conda run -n $env_name python make_video_from_results.py --results_dir results --multi_panel --output multi_agent_panel.mp4
    else
        echo "[ERROR] make_video_from_results.py not found!"
        exit 1
    fi

else
    echo "[INFO] Conda not found. Using python venv."
    python$PYTHON_VERSION -m venv $env_name
    source $env_name/bin/activate
    if [ -f requirements.txt ]; then
        echo "[INFO] Installing dependencies from requirements.txt using pip"
        pip install --upgrade pip
        pip install -r requirements.txt
    else
        echo "[ERROR] requirements.txt not found!"
        exit 1
    fi
    if [ -f run.py ]; then
        echo "[INFO] Running run.py simulation..."
        python run.py
    else
        echo "[ERROR] run.py not found!"
        exit 1
    fi
    if [ -f make_video_from_results.py ]; then
        echo "[INFO] Generating multi-agent panel video..."
        python make_video_from_results.py --results_dir results --multi_panel --output multi_agent_panel.mp4
    else
        echo "[ERROR] make_video_from_results.py not found!"
        exit 1
    fi
fi

echo "[INFO] All steps completed successfully!" 