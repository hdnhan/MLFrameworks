FROM nvcr.io/nvidia/tensorrt:24.08-py3 AS binary
# Creating a non-root user
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1003

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo git \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# [Optional] Set the default user. Omit if you want to keep the default as root.
RUN mkdir -p /workspace && chown -R $USER_UID:$USER_GID /workspace
USER $USERNAME

# OpenCV
ENV OPENCV_INSTALL_DIR=/opencv
COPY --from=opencv-cpu $OPENCV_INSTALL_DIR $OPENCV_INSTALL_DIR

# To use opencv in python
RUN sudo apt-get install -y python3 python3-pip ffmpeg
RUN pip3 install numpy
RUN sudo ln -s $OPENCV_INSTALL_DIR/lib/python3.10/dist-packages/cv2 /usr/local/lib/python3.10/dist-packages/cv2

# To compile opencv with cpp code
RUN sudo -H pip3 install cmake
