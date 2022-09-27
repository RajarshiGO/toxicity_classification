from ubuntu
RUN apt update
RUN apt install -y vim curl apache2 apache2-utils python3 libapache2-mod-wsgi-py3 python3-pip
ADD ./requirements.txt /var/www/html 
WORKDIR /var/www/html 
RUN pip3 install -r requirements.txt
ADD ./config.conf /etc/apache2/sites-available/000-default.conf
EXPOSE 80
CMD ["apache2ctl", "-D", "FOREGROUND"]
# CMD /usr/bin/bash
