# Ryzen AI NPU Docker Environment – Quick Start

This document describes how to install the AMD Ryzen AI NPU driver, start the Docker-based runtime environment, and verify that the NPU is working correctly.

The workflow is **driver → docker environment → container → venv → quicktest**.

---

## 1. Install the NPU Driver (Host)

On the host system, install the AMD XDNA / NPU driver:

```bash
./install_xdna.sh
```

This step installs:

* XDNA kernel driver
* Required firmware
* Host-side runtime dependencies


## 2. Detect Environment and Build Docker Setup

Run the build script on the host:

```bash
./build.sh
```

This script will:

* Check host prerequisites (Docker, XRT, runtime libraries)
* Pull the official Ryzen AI NPU Docker image
* Tag the image as `ryzen_ai_npu:latest`
* Start the container via **Docker Compose**

After completion, you should see messages indicating:

```text
Ryzen AI NPU container is running
NPU runtime environment is ready
```

Verify the container is up:

```bash
docker ps
```

You should see:

```text
ryzen-ai-npu
```

---

## 3. Enter the Docker Container

Enter the running container using:

```bash
docker exec -it ryzen-ai-npu bash
```

You should now be inside the container shell, for example:

```text
root@<host>:/workspace#
```

---

## 4. Activate the Python Virtual Environment

Inside the container, activate the Ryzen AI Python virtual environment:

```bash
source /home/ryzen_ai_1.6/venv/bin/activate
```

After activation, the shell prompt should change to:

```text
(ryzen-ai) root@<host>:/workspace#
```

> Note:
> The virtual environment is **not activated automatically** when using `docker exec`.
> This is expected Docker behavior.

---

## 5. Run the NPU Verification Test

Run the built-in quick test to verify NPU functionality:

```bash
cd /home/ryzen_ai_1.6/venv/quicktest
python quicktest.py
```

If the setup is correct, you should observe:

* One or more subgraphs assigned to the NPU
* Final output similar to:

```text
Test Passed
```

This confirms that:

* The NPU kernel driver is working
* XRT is correctly visible inside the container
* The Ryzen AI runtime can execute workloads on the NPU



## 6. Run YOLO Object Detection on the NPU

After verifying the basic NPU functionality, you can run advanced applications like real-time object detection using YOLO models.

Ensure you are still inside the container and the virtual environment is activated (steps 3-4). Then navigate to the appropriate directory:

```bash
cd /workspace
```
Run the YOLO object detection script with NPU acceleration:

```bash
python3 ./src/advantech-yolo-NPU.py \
  --model ./models/yolo11n.onnx \
  --video-file ./data/test.mp4 \
  --save-video \
  --output ./results/test_out.mp4 \
  --no-display
```

Command Parameters Explained:
Parameter	Description
--model ./models/yolo11n.onnx	Path to the YOLO model in ONNX format
--video-file ./data/test.mp4	Input video file for processing
--save-video	Save the processed output video
--output ./results/test_out.mp4	Output path for the processed video
--no-display	Run in headless mode (no GUI display)

---

## Notes & Troubleshooting

* If the test falls back to CPU execution:

  * Check that required Boost runtime libraries are installed
  * Ensure `/opt/xilinx/xrt` exists and is mounted into the container
* The Docker container **reuses host-installed XRT and firmware**
* Reboot the host after any NPU driver or firmware update

---

## Summary

1. Install NPU driver on the host
2. Run `build.sh` to prepare and start the Docker environment
3. Enter the container with `docker exec`
4. Activate the Python virtual environment
5. Run `quicktest.py` to verify NPU execution

You are now ready to run Ryzen AI workloads inside Docker using the NPU.
