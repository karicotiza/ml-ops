#!/bin/bash

sleep 15

if ! psql -U "$POSTGRES_USER" -d postgres -tc "SELECT 1 FROM pg_database WHERE datname = 'airflow'" | grep -q 1; then
    psql -U "$POSTGRES_USER" -d postgres -c "CREATE DATABASE airflow"
    echo "Created airflow database"
else
    echo "Database airflow already exists"
fi

if ! psql -U "$POSTGRES_USER" -d postgres -tc "SELECT 1 FROM pg_database WHERE datname = 'feast'" | grep -q 1; then
    psql -U "$POSTGRES_USER" -d postgres -c "CREATE DATABASE feast"
    echo "Created feast database"
else
    echo "Database feast already exists"
fi
