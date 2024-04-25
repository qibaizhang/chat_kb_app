from waitress import serve
from app import app

serve(app.server, host='localhost', port=8055)