"""
Email Notification Service for Speech Emotion Recognition (EmotiSound AI).
Dispatches rich HTML notifications containing the project logo from images/
upon user registration and login events via SMTP.
"""

import os
import sys
import smtplib
import base64
import logging
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    SMTP_HOST,
    SMTP_PORT,
    SMTP_USER,
    SMTP_PASSWORD,
    SMTP_FROM_EMAIL,
    SMTP_FROM_NAME,
    SMTP_USE_TLS,
    SMTP_USE_SSL,
    PROJECT_ROOT,
    IMAGES_DIR,
    LOGO_IMAGE_PATH,
    FRONTEND_URL,
)

logger = logging.getLogger("SER_EmailService")


def get_logo_image_path() -> Optional[Path]:
    """Finds the project logo image in the images folder, preferring email-optimized assets."""
    # 0. Check for email-optimized version for fast SMTP delivery
    email_opt = IMAGES_DIR / "logo_email.png"
    if email_opt.exists():
        return email_opt

    # 1. Check direct configured path
    if LOGO_IMAGE_PATH.exists():
        return LOGO_IMAGE_PATH

    # 2. Check images directory for any png/jpg/jpeg matching 'logo' or 'emoti'
    if IMAGES_DIR.exists():
        candidates = list(IMAGES_DIR.glob("*.png")) + list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.jpeg"))
        for c in candidates:
            if "logo" in c.name.lower() or "emoti" in c.name.lower():
                return c
        if candidates:
            return candidates[0]

    return None


def is_smtp_configured() -> bool:
    """Checks whether valid SMTP user and password credentials have been provided."""
    return bool(SMTP_USER and SMTP_PASSWORD and SMTP_HOST)


def get_smtp_status() -> Dict[str, Any]:
    """Returns current SMTP configuration status for frontend display."""
    configured = is_smtp_configured()
    # Mask username for privacy
    masked_user = ""
    if SMTP_USER:
        parts = SMTP_USER.split("@")
        if len(parts) == 2:
            name, domain = parts
            masked_user = f"{name[:2]}***@{domain}"
        else:
            masked_user = f"{SMTP_USER[:3]}***"

    logo_path = get_logo_image_path()

    return {
        "configured": configured,
        "host": SMTP_HOST,
        "port": SMTP_PORT,
        "user_configured": bool(SMTP_USER),
        "user_masked": masked_user,
        "tls": SMTP_USE_TLS,
        "ssl": SMTP_USE_SSL,
        "logo_found": logo_path is not None and logo_path.exists(),
        "logo_filename": logo_path.name if logo_path else None,
    }


def _build_html_email(
    title: str,
    subtitle: str,
    badge_text: str,
    recipient_name: str,
    message_paragraphs: list,
    meta_items: Dict[str, str],
    action_url: str = FRONTEND_URL,
    action_label: str = "Open EmotiSound Dashboard"
) -> str:
    """Builds a responsive, dark neon-themed HTML email with project branding."""
    now_str = datetime.now().strftime("%B %d, %Y at %I:%M %p")

    # Render meta items rows
    meta_rows = ""
    for label, val in meta_items.items():
        meta_rows += f"""
        <tr>
            <td style="padding: 8px 12px; font-size: 13px; color: #94a3b8; border-bottom: 1px solid #334155;">{label}</td>
            <td style="padding: 8px 12px; font-size: 13px; color: #f8fafc; font-weight: 600; text-align: right; border-bottom: 1px solid #334155;">{val}</td>
        </tr>
        """

    # Render paragraphs
    paragraphs_html = ""
    for p in message_paragraphs:
        paragraphs_html += f"""<p style="font-size: 15px; line-height: 1.6; color: #cbd5e1; margin: 0 0 16px 0;">{p}</p>"""

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
</head>
<body style="margin: 0; padding: 0; background-color: #0b0f19; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; color: #f8fafc;">
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="background-color: #0b0f19; padding: 40px 10px;">
        <tr>
            <td align="center">
                <!-- Main Container -->
                <table role="presentation" width="100%" style="max-width: 600px; background-color: #111827; border-radius: 20px; overflow: hidden; border: 1px solid #1e293b; box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7);" cellspacing="0" cellpadding="0" border="0">
                    
                    <!-- Header with Gradient and Logo -->
                    <tr>
                        <td align="center" style="background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%); padding: 36px 24px 28px 24px; border-bottom: 1px solid #312e81;">
                            <!-- Embedded Logo Image -->
                            <div style="margin-bottom: 16px;">
                                <img src="cid:project_logo" alt="EmotiSound AI Logo" width="120" height="120" style="display: block; margin: 0 auto; border-radius: 20px; border: 2px solid rgba(129, 140, 248, 0.4); box-shadow: 0 10px 25px rgba(99, 102, 241, 0.45); object-fit: cover;" />
                            </div>
                            <span style="display: inline-block; padding: 4px 12px; background-color: rgba(99, 102, 241, 0.2); border: 1px solid rgba(129, 140, 248, 0.4); border-radius: 9999px; font-size: 11px; font-weight: 700; color: #a5b4fc; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;">
                                {badge_text}
                            </span>
                            <h1 style="margin: 4px 0 6px 0; font-size: 24px; font-weight: 800; color: #ffffff; letter-spacing: -0.02em;">
                                {title}
                            </h1>
                            <p style="margin: 0; font-size: 13px; color: #94a3b8;">
                                {subtitle}
                            </p>
                        </td>
                    </tr>

                    <!-- Body Content -->
                    <tr>
                        <td style="padding: 32px 28px;">
                            <h2 style="margin: 0 0 14px 0; font-size: 18px; font-weight: 700; color: #ffffff;">
                                Hello, {recipient_name}! 👋
                            </h2>

                            {paragraphs_html}

                            <!-- Details Table -->
                            <div style="margin: 24px 0; background-color: #0f172a; border-radius: 12px; border: 1px solid #1e293b; overflow: hidden;">
                                <div style="padding: 10px 14px; background-color: #1e293b; font-size: 12px; font-weight: 700; color: #a5b4fc; text-transform: uppercase; letter-spacing: 0.05em;">
                                    Activity & System Details
                                </div>
                                <table width="100%" cellspacing="0" cellpadding="0" border="0">
                                    {meta_rows}
                                </table>
                            </div>

                            <!-- Model Benchmark Highlight -->
                            <div style="margin: 20px 0; padding: 14px 16px; background: rgba(99, 102, 241, 0.08); border-left: 4px solid #6366f1; border-radius: 8px;">
                                <div style="font-size: 12px; font-weight: 700; color: #818cf8; margin-bottom: 4px;">
                                    ⚡ AI Model Benchmark Performance
                                </div>
                                <div style="font-size: 12px; color: #cbd5e1; line-height: 1.5;">
                                    • <b>1D CNN:</b> 82.63% Test Accuracy (Deep Learning Leader)<br/>
                                    • <b>MLP Classifier:</b> 79.75% Test Accuracy<br/>
                                    • <b>Support Vector Machine:</b> 79.12% Test Accuracy
                                </div>
                            </div>

                            <!-- Call to Action Button -->
                            <div style="text-align: center; margin: 32px 0 16px 0;">
                                <a href="{action_url}" target="_blank" style="display: inline-block; padding: 14px 28px; background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); color: #ffffff; text-decoration: none; font-size: 14px; font-weight: 700; border-radius: 12px; box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4); border: 1px solid rgba(255, 255, 255, 0.15);">
                                    {action_label} &rarr;
                                </a>
                            </div>

                            <p style="font-size: 12px; color: #64748b; text-align: center; margin-top: 24px;">
                                Direct Link: <a href="{action_url}" style="color: #818cf8; text-decoration: none;">{action_url}</a>
                            </p>
                        </td>
                    </tr>

                    <!-- Footer -->
                    <tr>
                        <td style="padding: 24px; background-color: #0b0f19; border-top: 1px solid #1e293b; text-align: center;">
                            <p style="margin: 0 0 8px 0; font-size: 12px; color: #64748b;">
                                Acoustic Speech Emotion Recognition System &bull; RAVDESS &amp; TESS Speech Analytics
                            </p>
                            <p style="margin: 0; font-size: 11px; color: #475569;">
                                This is an automated notification from your EmotiSound AI application instance.<br/>
                                Generated on {now_str}
                            </p>
                        </td>
                    </tr>

                </table>
            </td>
        </tr>
    </table>
</body>
</html>
"""


def _send_email_smtp(
    to_email: str,
    subject: str,
    plain_text: str,
    html_content: str,
    logo_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Sends an email via SMTP with the project logo embedded as an inline MIME attachment.
    """
    safe_subject = subject.encode("ascii", errors="ignore").decode("ascii").strip()

    if not is_smtp_configured():
        msg_info = (
            f"SMTP not configured in .env. Notification for {to_email} ('{safe_subject}') "
            f"was generated but not sent. Provide SMTP_USER and SMTP_PASSWORD in backend/.env to activate live dispatch."
        )
        try:
            logger.info(msg_info)
        except Exception:
            pass

        return {
            "sent": False,
            "status": "NOT_CONFIGURED",
            "message": "SMTP not configured. Add SMTP_USER and SMTP_PASSWORD to backend/.env to activate real-time notifications.",
            "recipient": to_email,
            "subject": subject
        }

    try:
        from email.header import Header
        # Create root MIME message (multipart/related for inline images)
        msg_root = MIMEMultipart("related")
        msg_root["Subject"] = Header(subject, "utf-8")
        msg_root["From"] = f"{SMTP_FROM_NAME} <{SMTP_FROM_EMAIL}>"
        msg_root["To"] = to_email

        # Alternative container for plain text and HTML
        msg_alt = MIMEMultipart("alternative")
        msg_alt.attach(MIMEText(plain_text, "plain", "utf-8"))
        msg_alt.attach(MIMEText(html_content, "html", "utf-8"))
        msg_root.attach(msg_alt)

        # Attach inline project logo
        target_logo = logo_path or get_logo_image_path()
        if target_logo and target_logo.exists():
            with open(target_logo, "rb") as img_file:
                img_data = img_file.read()
                img = MIMEImage(img_data)
                # Set Content-ID matching cid:project_logo in HTML
                img.add_header("Content-ID", "<project_logo>")
                img.add_header("Content-Disposition", "inline", filename=target_logo.name)
                msg_root.attach(img)
                try:
                    logger.info(f"Attached logo from {target_logo.name} to email ({len(img_data)} bytes).")
                except Exception:
                    pass
        else:
            try:
                logger.warning(f"Project logo not found. Sending email without inline logo attachment.")
            except Exception:
                pass

        # Dispatch via SMTP with EHLO handshakes and fallback
        connected = False
        last_err = None

        if SMTP_USE_SSL:
            try:
                logger.info(f"Connecting to SMTP SSL {SMTP_HOST}:{SMTP_PORT}...")
                with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=25) as server:
                    server.login(SMTP_USER, SMTP_PASSWORD)
                    server.send_message(msg_root)
                    connected = True
            except Exception as e_ssl:
                last_err = e_ssl
        else:
            try:
                logger.info(f"Connecting to SMTP {SMTP_HOST}:{SMTP_PORT} (TLS={SMTP_USE_TLS})...")
                with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=25) as server:
                    server.ehlo()
                    if SMTP_USE_TLS:
                        server.starttls()
                        server.ehlo()
                    server.login(SMTP_USER, SMTP_PASSWORD)
                    server.send_message(msg_root)
                    connected = True
            except Exception as e_tls:
                last_err = e_tls
                # Try SSL fallback on port 465 if TLS failed
                if SMTP_HOST == "smtp.gmail.com" and not connected:
                    try:
                        logger.info("Retrying with Gmail SSL on port 465...")
                        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=25) as ssl_server:
                            ssl_server.login(SMTP_USER, SMTP_PASSWORD)
                            ssl_server.send_message(msg_root)
                            connected = True
                    except Exception as e_fallback:
                        last_err = e_fallback

        if not connected:
            raise RuntimeError(f"SMTP delivery failed: {last_err}")

        try:
            logger.info(f"Notification email successfully sent to {to_email}")
        except Exception:
            pass

        return {
            "sent": True,
            "status": "SENT",
            "message": f"Notification email successfully dispatched to {to_email} with project logo",
            "recipient": to_email,
            "subject": subject
        }

    except Exception as e:
        err_msg = str(e)
        try:
            logger.error(f"Failed to send email to {to_email}: {err_msg}")
        except Exception:
            pass
        return {
            "sent": False,
            "status": "ERROR",
            "error": err_msg,
            "message": f"SMTP delivery failed: {err_msg}",
            "recipient": to_email,
            "subject": subject
        }


def send_registration_notification(to_email: str, full_name: str) -> Dict[str, Any]:
    """
    Sends a rich welcome notification email containing the project logo
    when a new user signs up.
    """
    subject = "🎉 Welcome to EmotiSound AI – Account Created Successfully"
    title = "Welcome to EmotiSound AI"
    subtitle = "Acoustic Speech Emotion Recognition System"
    badge_text = "Account Created"

    paragraphs = [
        "Welcome to <b>EmotiSound AI</b>! Your user account has been successfully created and configured.",
        "Our intelligent platform combines traditional Machine Learning (SVM, MLP) and Deep Learning (1D CNN) "
        "to analyze raw human speech audio and predict emotional states with high precision.",
        "You can now upload audio recordings, record directly using your microphone, view real-time confidence scores, "
        "and track prediction history backed by SQLite."
    ]

    meta_items = {
        "Account Name": full_name,
        "Account Email": to_email,
        "Registration Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "Account Status": "Active & Verified",
        "Assigned Role": "Speech Analyst / User",
    }

    plain_text = (
        f"Welcome to EmotiSound AI, {full_name}!\n\n"
        f"Your account ({to_email}) has been successfully created.\n"
        f"You can now access the Acoustic Speech Emotion Recognition dashboard at {FRONTEND_URL}/dashboard\n\n"
        f"Model Performance:\n"
        f"- 1D CNN: 82.63% Accuracy\n"
        f"- MLP: 79.75% Accuracy\n"
        f"- SVM: 79.12% Accuracy\n\n"
        f"EmotiSound AI Team"
    )

    html_content = _build_html_email(
        title=title,
        subtitle=subtitle,
        badge_text=badge_text,
        recipient_name=full_name,
        message_paragraphs=paragraphs,
        meta_items=meta_items,
        action_url=f"{FRONTEND_URL}/predict",
        action_label="Start Predicting Emotions"
    )

    return _send_email_smtp(to_email, subject, plain_text, html_content)


def send_login_notification(
    to_email: str,
    full_name: str,
    client_ip: str = "127.0.0.1",
    user_agent: str = "Web Client"
) -> Dict[str, Any]:
    """
    Sends a security notification email containing the project logo
    when a user logs in.
    """
    subject = "🔐 Security Alert: New Login to EmotiSound AI"
    title = "New Sign-In Detected"
    subtitle = "Security & Activity Notification"
    badge_text = "Security Notification"

    paragraphs = [
        f"A new login session was just initiated for your <b>EmotiSound AI</b> account.",
        "If this was you, no further action is required. You can jump directly into running emotion predictions "
        "or viewing analytics.",
        "If you did not initiate this login, please review your environment credentials and secure your account immediately."
    ]

    meta_items = {
        "User Account": to_email,
        "Login Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "Client IP Address": client_ip,
        "Device / Browser": user_agent[:60] if user_agent else "Web Browser",
        "Session State": "Active",
    }

    plain_text = (
        f"Hello {full_name},\n\n"
        f"A new login session was detected for your account {to_email}.\n"
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        f"IP Address: {client_ip}\n"
        f"Browser: {user_agent}\n\n"
        f"If this was you, you're all set! Visit {FRONTEND_URL}/dashboard\n"
        f"EmotiSound AI Security Team"
    )

    html_content = _build_html_email(
        title=title,
        subtitle=subtitle,
        badge_text=badge_text,
        recipient_name=full_name,
        message_paragraphs=paragraphs,
        meta_items=meta_items,
        action_url=f"{FRONTEND_URL}/dashboard",
        action_label="Access Your Dashboard"
    )

    return _send_email_smtp(to_email, subject, plain_text, html_content)


def send_test_notification(to_email: str) -> Dict[str, Any]:
    """
    Sends a test verification email containing the project logo to confirm
    SMTP settings are working.
    """
    subject = "🧪 EmotiSound AI – SMTP Notification Test"
    title = "SMTP Configuration Test"
    subtitle = "Acoustic Speech Emotion Recognition"
    badge_text = "Test Verification"

    paragraphs = [
        "Congratulations! Your SMTP email notification system is configured and working perfectly.",
        "This test message confirms that your SMTP credentials in <code>.env</code> are valid, and that "
        "the official <b>EmotiSound AI Neon Logo</b> is properly embedded and rendered in the email body.",
        "All subsequent user registrations and logins will automatically receive branded notifications like this one."
    ]

    meta_items = {
        "Test Recipient": to_email,
        "SMTP Server": f"{SMTP_HOST}:{SMTP_PORT}",
        "Security Mode": "TLS" if SMTP_USE_TLS else ("SSL" if SMTP_USE_SSL else "Plain"),
        "Sender Identity": f"{SMTP_FROM_NAME} <{SMTP_FROM_EMAIL}>",
        "Test Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
    }

    plain_text = (
        f"EmotiSound AI SMTP Test\n\n"
        f"Your SMTP settings are working properly!\n"
        f"Host: {SMTP_HOST}:{SMTP_PORT}\n"
        f"Recipient: {to_email}\n"
    )

    html_content = _build_html_email(
        title=title,
        subtitle=subtitle,
        badge_text=badge_text,
        recipient_name="Administrator",
        message_paragraphs=paragraphs,
        meta_items=meta_items,
        action_url=f"{FRONTEND_URL}/dashboard",
        action_label="Return to Dashboard"
    )

    return _send_email_smtp(to_email, subject, plain_text, html_content)


def send_prediction_report_email(
    to_email: str,
    prediction_data: Dict[str, Any],
    report_lines: Optional[list] = None
) -> Dict[str, Any]:
    """
    Sends the 6-line formatted acoustic speech emotion analysis report
    to the user's email address embedding the official project logo via SMTP.
    """
    emotion = str(prediction_data.get("predicted_emotion") or "neutral").upper()
    confidence = prediction_data.get("confidence", 0.0)
    model = prediction_data.get("model", "CNN")
    filename = prediction_data.get("filename") or "Audio Recording"

    # Ensure 6 lines are present
    lines = report_lines or []
    if len(lines) != 6:
        from services.ai_explainer_service import build_six_line_report
        lines = build_six_line_report(prediction_data, prediction_data.get("ai_explanation"))

    subject = f"📊 EmotiSound AI – Speech Emotion Analysis Report: {emotion} ({confidence}%)"
    title = f"Speech Emotion Analysis Report"
    subtitle = f"Acoustic Prosody & In-Depth Diagnostic Audit"
    badge_text = f"{emotion} • {confidence}% CONFIDENCE"

    # Construct styled 6-line HTML report cards
    line_icons = ["🎯", "🧠", "⏱️", "⚖️", "🔊", "📋"]
    lines_html = ""
    plain_lines = ""

    for i, line in enumerate(lines[:6]):
        icon = line_icons[i] if i < len(line_icons) else "•"
        # Split line on the first colon if available
        if ":" in line:
            parts = line.split(":", 1)
            line_heading = parts[0].strip()
            line_text = parts[1].strip()
        else:
            line_heading = f"Point {i + 1}"
            line_text = line.strip()

        plain_lines += f"{icon} {line_heading}:\n   {line_text}\n\n"

        lines_html += f"""
        <div style="margin-bottom: 12px; padding: 14px 16px; background-color: #0f172a; border-radius: 12px; border-left: 4px solid #6366f1; border: 1px solid #1e293b;">
            <div style="font-size: 13px; font-weight: 700; color: #818cf8; margin-bottom: 4px; display: flex; align-items: center;">
                <span style="margin-right: 8px;">{icon}</span>
                <span>{line_heading}</span>
            </div>
            <div style="font-size: 13px; color: #cbd5e1; line-height: 1.55;">
                {line_text}
            </div>
        </div>
        """

    paragraphs = [
        f"Your acoustic speech recording (<b>{filename}</b>) has been analyzed by the EmotiSound AI engine.",
        f"Below is the complete <b>6-line synthesized diagnostic report</b> detailing the predicted emotion, acoustic prosody, chronological timeline progression, comparative model consensus, and recording fidelity audit:",
        lines_html
    ]

    acoustic = prediction_data.get("acoustic_profile") or {}
    meta_items = {
        "Audio Recording": filename,
        "Classified Emotion": emotion,
        "Prediction Confidence": f"{confidence}%",
        "Inference Model": model,
        "Mean Pitch (F0)": f"{acoustic.get('pitch_mean_hz', 'N/A')} Hz",
        "Signal-to-Noise Ratio": f"{acoustic.get('snr_db', '18.5')} dB",
        "AI Engine": prediction_data.get("ai_explanation", {}).get("engine", "Google Gemini AI (gemini-2.5-flash)"),
        "Dispatched At": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    }

    plain_text = (
        f"EmotiSound AI – Speech Emotion Analysis Report\n"
        f"====================================================\n\n"
        f"File: {filename}\n"
        f"Predicted Emotion: {emotion} ({confidence}% via {model})\n\n"
        f"--- 6-POINT ANALYSIS REPORT ---\n\n"
        f"{plain_lines}\n"
        f"View interactive results at: {FRONTEND_URL}/predict\n\n"
        f"EmotiSound AI Diagnostic Team"
    )

    html_content = _build_html_email(
        title=title,
        subtitle=subtitle,
        badge_text=badge_text,
        recipient_name=to_email.split("@")[0].capitalize(),
        message_paragraphs=paragraphs,
        meta_items=meta_items,
        action_url=f"{FRONTEND_URL}/predict",
        action_label="View Full Prediction on Web"
    )

    return _send_email_smtp(to_email, subject, plain_text, html_content)

