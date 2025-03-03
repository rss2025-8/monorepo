# Start from ubuntu
FROM ubuntu:jammy

# Update so we can download packages
RUN apt-get update && apt-get upgrade -y
#Set the ROS distro
ENV ROS_DISTRO humble
ARG DEBIAN_FRONTEND=noninteractive


# add the ROS deb repo to the apt sources list
#
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		curl \
		wget \
		gnupg2 \
		lsb-release \
		ca-certificates \
        console-setup \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key  -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null


# Setup for ros
# Set up ROS
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-setuptools 

# 
# install ros2 packages
#
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		ros-${ROS_DISTRO}-desktop \
		python3-colcon-common-extensions \
        ros-dev-tools \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean



# Set up ROS2
RUN rosdep init
RUN rosdep update --include-eol-distros


# Install VNC and things to install noVNC
RUN apt-get update && apt-get install -y \
    tigervnc-standalone-server \
    wget \
    git \
    unzip

# Download NoVNC and unpack
ENV NO_VNC_VERSION 1.4.0
RUN wget -q https://github.com/novnc/noVNC/archive/v$NO_VNC_VERSION.zip
RUN unzip v$NO_VNC_VERSION.zip
RUN rm v$NO_VNC_VERSION.zip
RUN git clone https://github.com/novnc/websockify /noVNC-$NO_VNC_VERSION/utils/websockify

# Install a window manager
RUN apt-get update && apt-get install -y \
    openbox \
    x11-xserver-utils \
    xterm \
    dbus-x11



# Set up locales
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    locales \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
RUN locale-gen en_US en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV PYTHONIOENCODING=utf-8


# Install additional required packages for ROS
RUN apt update && apt install -y \
    ros-$ROS_DISTRO-tf2-geometry-msgs \
    ros-$ROS_DISTRO-ackermann-msgs \
    ros-$ROS_DISTRO-tf-transformations \
    ros-$ROS_DISTRO-navigation2 \
    ros-$ROS_DISTRO-xacro \
    ros-$ROS_DISTRO-joy \
    build-essential \
    cython3


# Install some cool programs
RUN apt update && apt install -y \
    sudo \
    vim \
    emacs \
    nano \
    gedit \
    screen \
    tmux \
    iputils-ping \
    feh

# Install helix :)
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:maveonair/helix-editor && apt-get update && apt-get install -y helix python3-pylsp black

# # Fix some ROS things
# run apt install -y \
#     python-pip \
#     ros-$ROS_DISTRO-compressed-image-transport \
#     libfreetype6-dev
# RUN pip install -U pip
# RUN pip install imutils
# RUN pip install -U matplotlib


# install additional ros things
RUN apt-get update && pip install transforms3d && pip install opencv-contrib-python



# Kill the bell!
RUN echo "set bell-style none" >> /etc/inputrc

# Copy in the entrypoint
COPY ./entrypoint.sh /usr/bin/entrypoint.sh
COPY ./xstartup.sh /usr/bin/xstartup.sh




# Create racecar_ws directory and src before switching to USER
ENV SIM_WS /home/sim_ws
RUN apt update
RUN mkdir -p $SIM_WS/src && cd $SIM_WS/src && git clone https://github.com/Sebastian-Garcia/racecar_simulator.git
RUN /bin/bash -c 'source /opt/ros/$ROS_DISTRO/setup.bash; cd $SIM_WS; colcon build;'



# Copy in default config files
COPY ./config/bash.bashrc /etc/
COPY ./config/screenrc /etc/
COPY ./config/vimrc /etc/vim/vimrc
ADD ./config/openbox /etc/X11/openbox/
COPY ./config/XTerm /etc/X11/app-defaults/
COPY ./config/default.rviz /tmp/

# Create a user
RUN useradd -ms /bin/bash racecar
RUN echo 'racecar:racecar@mit' | chpasswd
RUN usermod -aG sudo racecar
USER racecar
WORKDIR /home/racecar

COPY ./entrypoint.sh /usr/bin/entrypoint.sh
COPY ./xstartup.sh /usr/bin/xstartup.sh
