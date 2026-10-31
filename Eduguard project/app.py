import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import smtplib
from email.message import EmailMessage
from datetime import datetime
import io
import html
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EduGuard AI",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

# HOD email for this presentation/demo
HOD_EMAIL = "hodaidskpriet@gmail.com"

# Email settings:
# Put the sender Gmail address and Gmail App Password in
# Streamlit secrets. Do NOT put your real password in this file.
#
# .streamlit/secrets.toml
#
# EMAIL_ADDRESS = "yourgmail@gmail.com"
# EMAIL_APP_PASSWORD = "your-16-character-app-password"
#
# The HOD can be the receiver. The sender account is used only
# to send the notification.


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_score",
    "learning_activity",
    "previous_gpa"
]

TARGET = "risk"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("students.csv")


# ============================================================
# TRAIN RANDOM FOREST MODEL
# ============================================================

@st.cache_resource
def train_model(data):

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)

    return model, accuracy, X_test, y_test, pred


# ============================================================
# FACULTY NOTIFICATION ENGINE
# ============================================================

def generate_alerts(data, model):

    alerts = []

    for _, row in data.iterrows():

        X_student = pd.DataFrame([
            row[FEATURES].to_dict()
        ])

        prediction = model.predict(X_student)[0]

        probabilities = model.predict_proba(X_student)[0]
        classes = list(model.classes_)

        probability = probabilities[
            classes.index(prediction)
        ]

        # Create notification for Medium and High Risk
        if prediction in ["Medium Risk", "High Risk"]:

            factors = []

            if row["attendance"] < 70:
                factors.append("Low attendance")

            if row["internal_marks"] < 50:
                factors.append("Low internal marks")

            if row["assignment_score"] < 60:
                factors.append("Low assignment performance")

            if row["learning_activity"] < 50:
                factors.append("Low learning activity")

            if row["previous_gpa"] < 6:
                factors.append("Low previous GPA")

            if not factors:
                factors.append("Multiple academic indicators")

            alerts.append({
                "student_id": row["student_id"],
                "risk_level": prediction,
                "probability": probability,
                "message": (
                    f"Student {row['student_id']} has reached "
                    f"{prediction.upper()} level."
                ),
                "factors": factors
            })

    # High Risk notifications appear first
    alerts.sort(
        key=lambda x: (
            0 if x["risk_level"] == "High Risk" else 1,
            -x["probability"]
        )
    )

    return alerts


# ============================================================
# EMAIL FUNCTIONS
# ============================================================

def get_email_settings():
    sender_email = os.getenv("EDUGUARD_EMAIL", "")
    app_password = os.getenv("EDUGUARD_APP_PASSWORD", "")
    return sender_email, app_password


def send_hod_email(student_id, prediction, probability, factors, row):
    """
    Sends a high-risk notification to the HOD.
    The HOD email is fixed for this presentation:
    saadhanaanandhakumar@gmail.com
    """

    sender_email, app_password = get_email_settings()

    if not sender_email or not app_password:
        return False, (
            "Email credentials are not configured. "
            "Add EMAIL_ADDRESS and EMAIL_APP_PASSWORD "
            "to Streamlit secrets."
        )

    subject = (
        f"🚨 EduGuard AI - HIGH RISK Student Alert - {student_id}"
    )

    factors_text = "\n".join(
        [f"• {factor}" for factor in factors]
    )

    body = f"""
Dear HOD,

EduGuard AI has detected a HIGH RISK student.

Student ID: {student_id}
Risk Level: {prediction}
Prediction Confidence: {probability * 100:.1f}%

Student Performance:
- Attendance: {row['attendance']}%
- Internal Marks: {row['internal_marks']}
- Assignment Score: {row['assignment_score']}%
- Learning Activity: {row['learning_activity']}%
- Previous GPA: {row['previous_gpa']}

Warning Factors:
{factors_text}

Recommended Action:
- Faculty review
- Follow up with the student
- Provide appropriate academic support
- Monitor the student's progress

This notification was generated by the EduGuard AI demo system.

Regards,
EduGuard AI
"""

    try:
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = sender_email
        message["To"] = HOD_EMAIL
        message.set_content(body)

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, app_password)
            server.send_message(message)

        return True, f"Email successfully sent to {HOD_EMAIL}"

    except Exception as e:
        return False, f"Email sending failed: {str(e)}"


# ============================================================
# REPORT FUNCTIONS
# ============================================================

def create_risk_report(data, model):
    """
    Creates a complete CSV report containing each student's
    model-predicted risk and prediction confidence.
    """

    report = data.copy()

    predictions = []
    confidences = []

    for _, row in data.iterrows():

        X_student = pd.DataFrame([
            row[FEATURES].to_dict()
        ])

        prediction = model.predict(X_student)[0]
        probabilities = model.predict_proba(X_student)[0]
        classes = list(model.classes_)

        confidence = probabilities[
            classes.index(prediction)
        ]

        predictions.append(prediction)
        confidences.append(round(confidence * 100, 2))

    report["AI_Predicted_Risk"] = predictions
    report["Prediction_Confidence_%"] = confidences

    return report


def create_student_report(row, prediction, probability, reasons):
    """
    Creates a simple HTML report for one student.
    """

    reasons_html = "".join(
        [f"<li>{html.escape(str(x))}</li>" for x in reasons]
    )

    risk = html.escape(str(prediction))
    sid = html.escape(str(row["student_id"]))

    report_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>EduGuard AI Student Report - {sid}</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 40px;
    background: #f5f7fb;
}}
.card {{
    background: white;
    padding: 25px;
    border-radius: 12px;
    margin-bottom: 20px;
}}
h1, h2 {{
    color: #222;
}}
table {{
    width: 100%;
    border-collapse: collapse;
}}
th, td {{
    border: 1px solid #ddd;
    padding: 10px;
    text-align: left;
}}
th {{
    background: #f0f2f6;
}}
.risk {{
    font-size: 24px;
    font-weight: bold;
}}
</style>
</head>

<body>

<div class="card">
<h1>🎓 EduGuard AI</h1>
<h2>Individual Student Risk Report</h2>
<p><b>Generated:</b> {datetime.now().strftime("%d-%m-%Y %H:%M:%S")}</p>
</div>

<div class="card">
<h2>Student Details</h2>
<p><b>Student ID:</b> {sid}</p>
<p class="risk">Risk Level: {risk}</p>
<p><b>Prediction Confidence:</b> {probability * 100:.1f}%</p>
</div>

<div class="card">
<h2>Academic Performance</h2>
<table>
<tr><th>Feature</th><th>Value</th></tr>
<tr><td>Attendance</td><td>{row["attendance"]}%</td></tr>
<tr><td>Internal Marks</td><td>{row["internal_marks"]}</td></tr>
<tr><td>Assignment Score</td><td>{row["assignment_score"]}%</td></tr>
<tr><td>Learning Activity</td><td>{row["learning_activity"]}%</td></tr>
<tr><td>Previous GPA</td><td>{row["previous_gpa"]}</td></tr>
</table>
</div>

<div class="card">
<h2>Warning Factors</h2>
<ul>
{reasons_html}
</ul>
</div>

<div class="card">
<h2>Recommended Intervention</h2>
<ul>
<li>Faculty review recommended</li>
<li>Follow up on attendance and academic performance</li>
<li>Provide appropriate academic support</li>
<li>Monitor progress after intervention</li>
</ul>
</div>

</body>
</html>
"""

    return report_html


# ============================================================
# RISK ICON
# ============================================================

def risk_icon(label):

    return {
        "Low Risk": "🟢",
        "Medium Risk": "🟡",
        "High Risk": "🔴"
    }.get(label, "⚪")


# ============================================================
# INTERVENTION RECOMMENDATIONS
# ============================================================

def intervention(label):

    if label == "High Risk":

        return [
            "Faculty review recommended",
            "Follow up on attendance",
            "Provide additional academic support",
            "Monitor assignment completion",
            "Reassess performance after a defined period"
        ]

    if label == "Medium Risk":

        return [
            "Monitor attendance and assessment trends",
            "Encourage assignment completion",
            "Consider targeted academic support"
        ]

    return [
        "Continue regular academic monitoring",
        "Encourage consistent learning activity"
    ]


# ============================================================
# LOAD DATA AND MODEL
# ============================================================

data = load_data()

model, accuracy, X_test, y_test, pred = train_model(data)

alerts = generate_alerts(data, model)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎓 EduGuard AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Student Analysis",
        "Notifications",
        "Reports",
        "Analytics",
        "Model Information"
    ]
)


# ============================================================
# NOTIFICATION COUNTS
# ============================================================

high_count = sum(
    a["risk_level"] == "High Risk"
    for a in alerts
)

medium_count = sum(
    a["risk_level"] == "Medium Risk"
    for a in alerts
)


# ============================================================
# SIDEBAR NOTIFICATION SUMMARY
# ============================================================

st.sidebar.markdown("---")

st.sidebar.subheader("🔔 Notifications")

st.sidebar.write(
    f"🔴 High Risk: **{high_count}**"
)

st.sidebar.write(
    f"🟡 Medium Risk: **{medium_count}**"
)

st.sidebar.write(
    f"📋 Total Alerts: **{len(alerts)}**"
)

st.sidebar.info(
    f"📧 HOD Email:\n{HOD_EMAIL}"
)


# ============================================================
# MAIN TITLE
# ============================================================

st.title("🎓 EduGuard AI")

st.caption(
    "AI-Based Student Performance Early Warning System"
)

st.info(
    "Demo prototype using synthetic student data. "
    "Predictions are for demonstration and should not be used "
    "as the sole basis for real academic decisions."
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.header("📊 Faculty Dashboard")

    counts = data["risk"].value_counts()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Students",
        len(data)
    )

    c2.metric(
        "🟢 Low Risk",
        int(counts.get("Low Risk", 0))
    )

    c3.metric(
        "🟡 Medium Risk",
        int(counts.get("Medium Risk", 0))
    )

    c4.metric(
        "🔴 High Risk",
        int(counts.get("High Risk", 0))
    )

    st.divider()

    st.subheader("Student Risk Overview")

    display = data[
        [
            "student_id",
            "attendance",
            "internal_marks",
            "assignment_score",
            "learning_activity",
            "previous_gpa",
            "risk"
        ]
    ].copy()

    display["risk"] = display["risk"].apply(
        lambda x: f"{risk_icon(x)} {x}"
    )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# STUDENT ANALYSIS
# ============================================================

elif page == "Student Analysis":

    st.header("👤 Individual Student Analysis")

    sid = st.selectbox(
        "Select Student",
        data["student_id"].tolist()
    )

    row = data[
        data["student_id"] == sid
    ].iloc[0]

    X_one = pd.DataFrame([
        row[FEATURES].to_dict()
    ])

    prediction = model.predict(X_one)[0]

    probabilities = model.predict_proba(X_one)[0]

    classes = list(model.classes_)

    prob = probabilities[
        classes.index(prediction)
    ]

    a, b, c, d = st.columns(4)

    a.metric(
        "Attendance",
        f"{row.attendance}%"
    )

    b.metric(
        "Internal Marks",
        f"{row.internal_marks}"
    )

    c.metric(
        "Assignment",
        f"{row.assignment_score}%"
    )

    d.metric(
        "Previous GPA",
        f"{row.previous_gpa}"
    )

    st.divider()

    st.subheader(
        f"{risk_icon(prediction)} {prediction}"
    )

    st.metric(
        "Prediction Confidence",
        f"{prob * 100:.1f}%"
    )

    # ========================================================
    # HIGH RISK WEBSITE NOTIFICATION
    # ========================================================

    if prediction == "High Risk":

        st.error(
            f"🚨 FACULTY ALERT: Student {sid} "
            "has reached HIGH RISK level."
        )

        st.warning(
            "The student requires faculty attention. "
            "Click the button below to send an email notification "
            "to the HOD."
        )

    elif prediction == "Medium Risk":

        st.warning(
            f"⚠️ FACULTY ALERT: Student {sid} "
            "has reached MEDIUM RISK level."
        )

    else:

        st.success(
            f"✅ Student {sid} is currently LOW RISK."
        )

    # ========================================================
    # WHY FLAGGED
    # ========================================================

    importances = pd.Series(
        model.feature_importances_,
        index=FEATURES
    ).sort_values(ascending=False)

    labels = {
        "attendance": "Attendance",
        "internal_marks": "Internal Marks",
        "assignment_score": "Assignment Score",
        "learning_activity": "Learning Activity",
        "previous_gpa": "Previous GPA"
    }

    reference = pd.Series({
        "attendance": 85,
        "internal_marks": 75,
        "assignment_score": 75,
        "learning_activity": 75,
        "previous_gpa": 8.0
    })

    values = X_one.iloc[0]

    concern = (
        (reference - values)
        .clip(lower=0)
        / reference
    ).fillna(0)

    contribution = (
        importances * concern
    ).sort_values(ascending=False)

    st.subheader("🔍 Why was this student flagged?")

    reasons = []

    for feat, score in contribution.items():

        if score > 0.02:

            reasons.append(
                (
                    labels[feat],
                    row[feat],
                    score
                )
            )

    reason_text = []

    if not reasons:

        st.success(
            "No major warning signal was identified "
            "by this demo explanation."
        )

        reason_text.append("Multiple academic indicators")

    else:

        for name, val, score in reasons[:4]:

            st.write(
                f"**{name}:** {val} — "
                "contributing warning signal"
            )

            reason_text.append(
                f"{name}: {val}"
            )

    st.subheader("📌 Suggested Intervention")

    for item in intervention(prediction):

        st.write(
            "• " + item
        )

    # ========================================================
    # SEND EMAIL BUTTON
    # ========================================================

    if prediction == "High Risk":

        st.divider()

        st.subheader("📧 HOD Notification")

        st.write(
            f"Email will be sent to: **{HOD_EMAIL}**"
        )

        if st.button(
            "🚨 Send High Risk Alert to HOD",
            type="primary",
            key=f"send_hod_{sid}"
        ):

            factors = [
                item for item in reason_text
            ]

            success, message = send_hod_email(
                student_id=sid,
                prediction=prediction,
                probability=prob,
                factors=factors,
                row=row
            )

            if success:
                st.success(
                    f"✅ {message}"
                )

                st.info(
                    f"📨 HOD notification sent for student {sid}."
                )

            else:
                st.error(
                    f"❌ {message}"
                )

    st.caption(
        "Explanation combines model feature importance "
        "with demo reference levels; it is not a "
        "high-stakes decision system."
    )


# ============================================================
# NOTIFICATIONS MODULE
# ============================================================

elif page == "Notifications":

    st.header("🔔 Faculty Notifications")

    st.caption(
        "Early warning alerts for students requiring "
        "academic attention."
    )

    if not alerts:

        st.success(
            "✅ No students have reached the warning level."
        )

    else:

        n1, n2, n3 = st.columns(3)

        n1.metric(
            "🔔 Total Alerts",
            len(alerts)
        )

        n2.metric(
            "🔴 High Risk",
            high_count
        )

        n3.metric(
            "🟡 Medium Risk",
            medium_count
        )

        st.divider()

        # ====================================================
        # HIGH RISK
        # ====================================================

        high_alerts = [
            a
            for a in alerts
            if a["risk_level"] == "High Risk"
        ]

        if high_alerts:

            st.subheader(
                "🚨 High Risk Notifications"
            )

            for alert in high_alerts:

                with st.container(border=True):

                    st.error(
                        f"🔴 HIGH RISK ALERT — "
                        f"{alert['student_id']}"
                    )

                    st.write(
                        f"**{alert['message']}**"
                    )

                    st.write(
                        f"Prediction confidence: "
                        f"**{alert['probability'] * 100:.1f}%**"
                    )

                    st.write(
                        "**Warning factors:** "
                        + ", ".join(
                            alert["factors"]
                        )
                    )

                    st.write(
                        "📌 **Recommended Action:** "
                        "Faculty review and timely "
                        "academic intervention."
                    )

                    # Faculty clicks this button to notify HOD.
                    sid = alert["student_id"]

                    student_row = data[
                        data["student_id"] == sid
                    ].iloc[0]

                    if st.button(
                        f"📧 Send High Risk Email to HOD - {sid}",
                        key=f"notification_email_{sid}"
                    ):

                        success, message = send_hod_email(
                            student_id=sid,
                            prediction=alert["risk_level"],
                            probability=alert["probability"],
                            factors=alert["factors"],
                            row=student_row
                        )

                        if success:
                            st.success(
                                f"✅ {message}"
                            )
                        else:
                            st.error(
                                f"❌ {message}"
                            )

        # ====================================================
        # MEDIUM RISK
        # ====================================================

        medium_alerts = [
            a
            for a in alerts
            if a["risk_level"] == "Medium Risk"
        ]

        if medium_alerts:

            st.subheader(
                "⚠️ Medium Risk Notifications"
            )

            for alert in medium_alerts:

                with st.container(border=True):

                    st.warning(
                        f"🟡 MEDIUM RISK ALERT — "
                        f"{alert['student_id']}"
                    )

                    st.write(
                        f"**{alert['message']}**"
                    )

                    st.write(
                        f"Prediction confidence: "
                        f"**{alert['probability'] * 100:.1f}%**"
                    )

                    st.write(
                        "**Warning factors:** "
                        + ", ".join(
                            alert["factors"]
                        )
                    )

                    st.write(
                        "📌 **Recommended Action:** "
                        "Monitor the student's academic progress."
                    )


# ============================================================
# REPORTS MODULE
# ============================================================

elif page == "Reports":

    st.header("📄 Student Reports")

    st.caption(
        "Generate and download student risk reports for the presentation."
    )

    # --------------------------------------------------------
    # Complete risk report
    # --------------------------------------------------------

    st.subheader("📊 Complete Student Risk Report")

    full_report = create_risk_report(data, model)

    csv_buffer = io.StringIO()
    full_report.to_csv(csv_buffer, index=False)

    st.download_button(
        label="⬇️ Download Complete Risk Report (CSV)",
        data=csv_buffer.getvalue(),
        file_name="EduGuard_AI_Risk_Report.csv",
        mime="text/csv"
    )

    st.dataframe(
        full_report,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --------------------------------------------------------
    # Individual student report
    # --------------------------------------------------------

    st.subheader("👤 Individual Student Report")

    report_sid = st.selectbox(
        "Select student for report",
        data["student_id"].tolist(),
        key="report_student"
    )

    report_row = data[
        data["student_id"] == report_sid
    ].iloc[0]

    report_X = pd.DataFrame([
        report_row[FEATURES].to_dict()
    ])

    report_prediction = model.predict(report_X)[0]

    report_probabilities = model.predict_proba(report_X)[0]
    report_classes = list(model.classes_)

    report_probability = report_probabilities[
        report_classes.index(report_prediction)
    ]

    # Generate warning factors
    report_factors = []

    if report_row["attendance"] < 70:
        report_factors.append("Low attendance")

    if report_row["internal_marks"] < 50:
        report_factors.append("Low internal marks")

    if report_row["assignment_score"] < 60:
        report_factors.append("Low assignment performance")

    if report_row["learning_activity"] < 50:
        report_factors.append("Low learning activity")

    if report_row["previous_gpa"] < 6:
        report_factors.append("Low previous GPA")

    if not report_factors:
        report_factors.append(
            "No major individual warning factor identified"
        )

    st.write(
        f"**Student:** {report_sid}"
    )

    st.write(
        f"**Risk:** {risk_icon(report_prediction)} "
        f"{report_prediction}"
    )

    st.write(
        f"**Confidence:** {report_probability * 100:.1f}%"
    )

    individual_html = create_student_report(
        report_row,
        report_prediction,
        report_probability,
        report_factors
    )

    st.download_button(
        label="⬇️ Download Individual Student Report (HTML)",
        data=individual_html,
        file_name=f"EduGuard_Report_{report_sid}.html",
        mime="text/html"
    )

    st.subheader("Report Preview")

    st.components.v1.html(
        individual_html,
        height=650,
        scrolling=True
    )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "Analytics":

    st.header("📈 Academic Analytics")

    c1, c2 = st.columns(2)

    with c1:

        st.subheader(
            "Risk Distribution"
        )

        st.bar_chart(
            data["risk"].value_counts()
        )

    with c2:

        st.subheader(
            "Attendance vs Internal Marks"
        )

        st.line_chart(
            data.set_index("student_id")[
                [
                    "attendance",
                    "internal_marks"
                ]
            ].head(30)
        )

    st.subheader(
        "Performance Trend — Demo Student"
    )

    sid = st.selectbox(
        "Choose student for trend",
        data["student_id"].tolist(),
        key="trend"
    )

    row = data[
        data.student_id == sid
    ].iloc[0]

    current = int(
        row.internal_marks
    )

    trend = pd.DataFrame({

        "Assessment": [
            "Test 1",
            "Test 2",
            "Test 3",
            "Test 4"
        ],

        "Marks": [
            min(100, current + 18),
            min(100, current + 10),
            min(100, current + 5),
            current
        ]

    }).set_index("Assessment")

    st.line_chart(trend)

    if trend["Marks"].iloc[-1] < trend["Marks"].iloc[0] - 15:

        st.warning(
            "⚠️ Continuous decline detected "
            "in this demo trend."
        )

    else:

        st.success(
            "No strong decline detected "
            "in this demo trend."
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "Model Information":

    st.header("🤖 Model Information")

    st.write(
        "**Algorithm:** Random Forest Classifier"
    )

    st.write(
        f"**Validation accuracy on the demo "
        f"test split:** {accuracy * 100:.1f}%"
    )

    st.write(
        "**Features:** Attendance, internal marks, "
        "assignment score, learning activity, previous GPA"
    )

    st.write(
        "**Target:** Low Risk / Medium Risk / High Risk"
    )

    st.warning(
        "This project uses synthetic data generated "
        "for a hackathon prototype. The displayed "
        "accuracy is not evidence of real-world performance."
    )

    st.subheader(
        "Feature Importance"
    )

    st.bar_chart(
        pd.Series(
            model.feature_importances_,
            index=FEATURES
        ).sort_values(ascending=False)
    )

    st.subheader(
        "Confusion Matrix"
    )

    cm = confusion_matrix(
        y_test,
        pred,
        labels=model.classes_
    )

    st.dataframe(
        pd.DataFrame(
            cm,
            index=model.classes_,
            columns=model.classes_
        ),
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.markdown("---")

st.sidebar.caption(
    "HackMint 26 prototype • HACM109"
)
