from app import create_app, db
from config.settings import config
import os

# Get configuration from environment
config_name = os.environ.get('FLASK_ENV', 'development')
app = create_app(config[config_name])

if __name__ == '__main__':
    # Note: Database tables already exist from sample data script
    # with app.app_context():
    #     db.create_all()
    
    # Run the application
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=app.config['DEBUG']
    )
