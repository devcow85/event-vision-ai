FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /app

# Install SSH server
RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /var/run/sshd

# Set root password
RUN echo 'root:abcd' | chpasswd

# Allow root login with password
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config


RUN git clone https://github.com/devcow85/event-vision-ai.git /app/event-vision-ai \
    && cd /app/event-vision-ai \
    && pip install .

# Expose SSH port
EXPOSE 22

# Start SSH service
CMD ["/usr/sbin/sshd", "-D"]