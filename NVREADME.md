NVIDIA Caffe
============

Caffe is a deep learning framework made with expression, speed, and modularity
in mind.  It was originally developed by the [Berkeley Vision and Learning
Center (BVLC)](http://caffe.berkeleyvision.org/) and by community contributors.

NVIDIA Caffe, also known as NVCaffe, is an NVIDIA-maintained fork of BVLC Caffe
tuned for NVIDIA GPUs, particularly in multi-GPU configurations.  It includes
multi-precision support as well as other NVIDIA-enhanced features and offers
performance specially tuned for the NVIDIA DGX-1.

## Contents of the NVIDIA Caffe container image

This image contains source and binaries for NVIDIA Caffe.  The pre-built
and installed version of NVIDIA Caffe is located in the
`/usr/local/[bin,share,lib]` directories.  The complete source code
is located in `/opt/caffe`.

This container image also includes pycaffe, which makes the Caffe interfaces
available for use through Python.

The NCCL library and NVCaffe bindings for NCCL are installed in this container,
and models using multiple GPUs will automatically leverage this library for
fast parallel training.

## Running NVIDIA Caffe

You can choose to use Caffe as provided by NVIDIA, or you can choose to
customize it.

NVIDIA Caffe can be run by issuing the `caffe train ...` command. For an
explanation of the command-line parameters, issue the `caffe --help` command.

## Customizing NVIDIA Caffe

You can customize NVIDIA Caffe one of two ways:

(1) Modify the version of the source code in this container and run your
customized version, or (2) use `docker build` to add your customizations on top
of this container if you want to add additional packages.

NVIDIA recommends option 2 for ease of migration to later versions of the
NVIDIA Caffe container image.

For more information, see https://docs.docker.com/engine/reference/builder/ for
a syntax reference.  Several example Dockerfiles are provided in the container
image in `/workspace/docker-examples`.

## Suggested Reading

For more information about Caffe, including tutorials, documentation, and
examples, see the [Caffe website]( http://caffe.berkeleyvision.org).  NVIDIA
Caffe typically utilizes the same input formats and configuration parameters as
Caffe, so community-authored materials regarding and pre-trained models for
Caffe usually also can be applied to NVIDIA Caffe.
