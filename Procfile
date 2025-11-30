# Procfile for Heroku, Railway, and similar platforms
web: gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 4 tep.dashboard_dash:server
