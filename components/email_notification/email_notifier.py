#!/usr/bin/env python3
"""
Email Notification Component for Kubeflow Pipeline
"""

import argparse
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

class EmailHelper:
    def __init__(self, sender_email, gmail_password):
        self.sender_email = sender_email
        self.gmail_password = gmail_password
        
    def send_email(self, sender, recipient, subject, body_text, body_html):
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = sender
            msg['To'] = recipient
            
            text_part = MIMEText(body_text, 'plain')
            html_part = MIMEText(body_html, 'html')
            
            msg.attach(text_part)
            msg.attach(html_part)
            
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(self.sender_email, self.gmail_password)
            server.sendmail(sender, recipient, msg.as_string())
            server.quit()
            
            print(f"Email sent successfully to {recipient}")
            return True
            
        except Exception as e:
            print(f"Failed to send email: {e}")
            return False

def create_email_body(action, pipeline_name, model_name="", metrics=""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    if action == "pipeline_complete":
        title = "✅ MLOps Pipeline Completed"
        message = f"The MLOps pipeline {pipeline_name} has completed successfully."
    elif action == "training_complete":
        title = "✅ Model Training Completed"
        message = f"Model training for {pipeline_name} has completed successfully."
    elif action == "model_registered":
        title = "✅ Model Registered"
        message = f"Model has been successfully registered in the model registry."
    else:
        title = "📧 Pipeline Notification"
        message = f"Notification from {pipeline_name} pipeline."
    
    metrics_html = ""
    if metrics:
        try:
            metrics_data = json.loads(metrics)
            metrics_html = "<h3> Model Metrics</h3><ul>"
            for k, v in metrics_data.items():
                if k != "model_type":
                    metrics_html += f"<li><b>{k}:</b> {v:.4f if isinstance(v, float) else v}</li>"
            metrics_html += "</ul>"
        except:
            metrics_html = f"<p><b>Metrics:</b> {metrics}</p>"
    
    body_html = f"""
        <html>
        <head>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 20px; }}
            .container {{ background-color: #ffffff; border-radius: 8px; padding: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            h1 {{ color: #4CAF50; }}
            .status-box {{ background-color: #e8f5e9; border-left: 5px solid #4CAF50; padding: 15px; margin: 20px 0; }}
        </style>
        </head>
        <body>
        <div class="container">
            <h1>{title}</h1>
            <p>{message}</p>
            
            <div class="status-box">
                <p><b>Pipeline:</b> {pipeline_name}</p>
                <p><b>Status:</b> <span style="color:green;">Completed Successfully</span></p>
                <p><b>Timestamp:</b> {timestamp}</p>
            </div>
            
            {metrics_html}
            
            <p>Regards,<br/>MLOps Platform</p>
        </div>
        </body>
        </html>
    """
    
    return body_html

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, required=True, help='Action type')
    parser.add_argument('--recipient-email', type=str, required=True, help='Recipient email')
    parser.add_argument('--sender-email', type=str, required=True, help='Sender email')
    parser.add_argument('--gmail-password', type=str, required=True, help='Gmail app password')
    parser.add_argument('--subject', type=str, required=True, help='Email subject')
    parser.add_argument('--message-content', type=str, required=True, help='Message content')
    parser.add_argument('--pipeline-name', type=str, required=True, help='Pipeline name')
    parser.add_argument('--model-name', type=str, default='', help='Model name')
    parser.add_argument('--model-metrics', type=str, default='', help='Model metrics')
    parser.add_argument('--error-details', type=str, default='', help='Error details')
    
    args = parser.parse_args()
    
    # Create email helper
    em = EmailHelper(args.sender_email, args.gmail_password)
    
    # Create HTML email content
    body_html = create_email_body(args.action, args.pipeline_name, args.model_name, args.model_metrics)
    
    # Send email
    em.send_email(
        sender=args.sender_email,
        recipient=args.recipient_email,
        subject=args.subject,
        body_text=args.message_content,
        body_html=body_html
    )
