FROM ubuntu
RUN apt update
RUN apt install -y apache2 apache2-utils python3 libapache2-mod-wsgi-py3 python3-pip
WORKDIR /var/www/html
RUN rm ./*
RUN mkdir templates
ADD ./requirements.txt ./ 
ADD ./app.py ./
ADD ./app.wsgi ./
ADD ./model.pth ./
ADD ./vocab.pt ./
ADD ./templates ./templates
RUN pip3 install -r requirements.txt
ADD ./config.conf /etc/apache2/sites-available/000-default.conf
ADD ./ports.conf /etc/apache2/ports.conf
CMD ["apache2ctl", "-D", "FOREGROUND"]
