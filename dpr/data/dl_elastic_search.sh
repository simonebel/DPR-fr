#!/bin/bash
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.6.2-linux-x86_64.tar.gz -q
tar -xzf elasticsearch-7.6.2-linux-x86_64.tar.gz
chown -R daemon:daemon elasticsearch-7.6.2
