variable "access_key" {}
variable "secret_key" {}
variable "proj_owner" {}
variable "aws_key_name" {}
variable "region" {
    default = "ap-northeast-2"
}
variable "ssh_cidr" {}
variable "ssh_key_file" {}


provider "aws" {
  access_key = "${var.access_key}"
  secret_key = "${var.secret_key}"
  region = "${var.region}"
}

resource "aws_default_subnet" "default" {
    availability_zone = "${var.region}"

    tags {
        Name = "Default subnet"
    }
}


resource "aws_security_group" "distper" {
    name = "distper-sg"
    description = "Security group for DistPER."

    ingress {
        from_port = 5557
        to_port = 5558
        protocol = "tcp"
        cidr_blocks = ["${aws_default_subnet.default.cidr_block}"]
    }

    # for ssh
    ingress {
        from_port = 22
        to_port = 22
        protocol = "tcp"
        cidr_blocks = ["${var.ssh_cidr}"]
    }

    egress {
        from_port = 0
        to_port = 0
        protocol = "-1"
        cidr_blocks = ["0.0.0.0/0"]
    }

    tags {
        Name = "distper-sg"
        Owner = "${var.proj_owner}"
    }
}

# task node
resource "aws_instance" "task" {
    ami = "ami-b9e357d7"          # Deep Learning AMI (Ubuntu) Version 11.0
    instance_type = "t2.micro"    # 4 Cores, 16 GiB RAM
    key_name = "${var.aws_key_name}"
    vpc_security_group_ids = ["${aws_security_group.distper.id}"]
    subnet_id = "${aws_default_subnet.default.id}"

    provisioner "remote-exec" {
        connection {
            type = "ssh"
            user = "ubuntu"
            private_key = "${file("${var.ssh_key_file}")}"
        }
        inline = [
            <<EOF
/home/ubuntu/anaconda3/bin/conda install -n pytorch_p36 -y opencv
/home/ubuntu/anaconda3/bin/conda install -n pytorch_p36 -y libprotobuf protobuf
git clone https://github.com/openai/gym
cd gym
/home/ubuntu/anaconda3/envs/pytorch_p36/bin/pip install -e .
/home/ubuntu/anaconda3/envs/pytorch_p36/bin/pip install gym[classic_control,atari]
cd
git clone https://github.com/haje01/distper.git
EOF
        ]
    }

    tags {
        Name = "distper-task"
        Owner = "${var.proj_owner}"
    }
}

# master node
resource "aws_instance" "master" {
    ami = "ami-b9e357d7"          # Deep Learning AMI (Ubuntu) Version 11.0
    # instance_type = "p2.xlarge"   # GPU, 4 Cores, 61 GiB RAM
    instance_type = "t2.micro"    # GPU, 4 Cores, 61 GiB RAM
    key_name = "${var.aws_key_name}"
    vpc_security_group_ids = ["${aws_security_group.distper.id}"]
    subnet_id = "${aws_default_subnet.default.id}"

    provisioner "remote-exec" {
        connection {
            type = "ssh"
            user = "ubuntu"
            private_key = "${file("${var.ssh_key_file}")}"
        }
        inline = [
            <<EOF
/home/ubuntu/anaconda3/bin/conda install -n pytorch_p36 -y scikit-image tensorflow tensorboard opencv
/home/ubuntu/anaconda3/envs/pytorch_p36/bin/pip install tensorboardX
/home/ubuntu/anaconda3/bin/conda install -n pytorch_p36 -y libprotobuf protobuf
git clone https://github.cloneom/openai/gym
cd gym
/home/ubuntu/anaconda3/envs/pytorch_p36/bin/pip install -e .
/home/ubuntu/anaconda3/envs/pytorch_p36/bin/pip install gym[classic_control,atari]
cd
git clone https://github.com/haje01/distper.git
export MASTER_IP=${aws_instance.master.ip}
EOF
        ]
    }

    tags {
        Name = "distper-master"
        Owner = "${var.proj_owner}"
    }
}
