import io
from datetime import datetime
import numpy as np, pandas as pd
import plotly.graph_objects as go, plotly.express as px
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, PageBreak, Image as RLImage, HRFlowable)
from reportlab.lib.enums import TA_CENTER

PAL = ["#58a6ff","#3fb950","#f78166","#bc8cff","#e3b341"]
def _c(i): return PAL[i % len(PAL)]
def _th(): return dict(plot_bgcolor="#0d1117",paper_bgcolor="#0d1117",font=dict(color="#c9d1d9",family="Inter"),
                        xaxis=dict(gridcolor="#21262d",linecolor="#21262d",tickfont=dict(color="#7d8590")),
                        yaxis=dict(gridcolor="#21262d",linecolor="#21262d",tickfont=dict(color="#7d8590")),
                        margin=dict(l=50,r=20,t=50,b=50))
def _img(fig, W, h=380):
    try:
        fig.update_layout(**_th()); png = fig.to_image(format="png",width=700,height=h,scale=2)
        return RLImage(io.BytesIO(png),width=W,height=W*h/700)
    except Exception as e:
        ss = getSampleStyleSheet()
        return Paragraph(f"Chart N/A: {e}", ParagraphStyle("e",parent=ss["Normal"],fontSize=8,textColor=colors.HexColor("#7d8590")))

def generate_pdf(session):
    buf = io.BytesIO(); W = A4[0]-4*cm
    doc = SimpleDocTemplate(buf,pagesize=A4,leftMargin=2*cm,rightMargin=2*cm,topMargin=2*cm,bottomMargin=2*cm)
    ss = getSampleStyleSheet()
    ST  = ParagraphStyle("T", parent=ss["Title"],   fontSize=20,textColor=colors.HexColor("#58a6ff"),spaceAfter=4,leading=26)
    SH1 = ParagraphStyle("H1",parent=ss["Heading1"],fontSize=13,textColor=colors.HexColor("#e6edf3"),spaceBefore=12,spaceAfter=4)
    SH2 = ParagraphStyle("H2",parent=ss["Heading2"],fontSize=10,textColor=colors.HexColor("#c9d1d9"),spaceBefore=6,spaceAfter=3)
    SB  = ParagraphStyle("B", parent=ss["Normal"],  fontSize=9, textColor=colors.HexColor("#8b949e"),leading=14,spaceAfter=4)
    SM  = ParagraphStyle("M", parent=ss["Normal"],  fontSize=8, fontName="Courier",textColor=colors.HexColor("#7d8590"),leading=12)
    SFT = ParagraphStyle("FT",parent=ss["Normal"],  fontSize=8, textColor=colors.HexColor("#7d8590"),alignment=TA_CENTER)
    def hr(): return HRFlowable(width="100%",thickness=0.4,color=colors.HexColor("#30363d"),spaceAfter=6,spaceBefore=6)
    def ts():
        return TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1c2128")),("TEXTCOLOR",(0,0),(-1,0),colors.HexColor("#58a6ff")),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,0),8),
            ("BOTTOMPADDING",(0,0),(-1,0),6),("TOPPADDING",(0,0),(-1,0),6),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#0d1117"),colors.HexColor("#161b22")]),
            ("TEXTCOLOR",(0,1),(-1,-1),colors.HexColor("#c9d1d9")),("FONTSIZE",(0,1),(-1,-1),8),
            ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#30363d")),
            ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
            ("TOPPADDING",(0,1),(-1,-1),4),("BOTTOMPADDING",(0,1),(-1,-1),4)])
    task = session["task"]; results = session["model_results"]; feats = session["feature_names"]
    story = [Spacer(1,.5*cm),Paragraph("CropML Research Studio",ST),Paragraph("ML Model Performance Report",SB),Spacer(1,.2*cm),hr()]
    meta = [["Session",session["name"]],["Dataset",session["dataset"]],["Task",task],
            ["Samples",f"{session['n_samples']:,}"],["Features",str(session["n_features"])],
            ["Target",session["target_col"]],["Test size",f"{int(session['test_size']*100)}%"],
            ["Models trained",str(len(results))],["Generated",datetime.now().strftime("%Y-%m-%d %H:%M")]]
    mt = Table([[Paragraph(r[0],SM),Paragraph(str(r[1]),SB)] for r in meta],colWidths=[3.5*cm,W-3.5*cm])
    mt.setStyle(ts()); story += [mt,PageBreak()]
    story += [Paragraph("Model Performance Summary",SH1),hr()]
    if task == "Regression":
        hdr = ["Model","R2","RMSE","MAE","Runtime(s)","Mem(MB)","Size(MB)"]
        rows = [hdr]+[[r["name"],f"{r.get('r2',0):.4f}",f"{r.get('rmse',0):.4f}",f"{r.get('mae',0):.4f}",
                       f"{r['runtime']:.3f}",f"{r['memory_mb']:.1f}",f"{r['model_size_mb']:.3f}"]
                      for r in sorted(results,key=lambda x:-x.get("r2",0))]
    else:
        hdr = ["Model","Accuracy","F1","Precision","Recall","AUC","Runtime(s)"]
        rows = [hdr]+[[r["name"],f"{r.get('accuracy',0):.4f}",f"{r.get('f1',0):.4f}",
                       f"{r.get('precision',0):.4f}",f"{r.get('recall',0):.4f}",
                       f"{r.get('auc',0) or 0:.4f}",f"{r['runtime']:.3f}"]
                      for r in sorted(results,key=lambda x:-x.get("accuracy",0))]
    cw = W/len(hdr); pt = Table(rows,colWidths=[cw]*len(hdr)); pt.setStyle(ts()); story += [pt,Spacer(1,.4*cm)]
    story += [Paragraph("Hyperparameter Configuration",SH1),hr()]
    for r in results:
        story.append(Paragraph(r["name"],SH2))
        if r["params"]:
            ph = [["Parameter","Value"]]+[[k,v] for k,v in r["params"].items()]
            ptbl = Table(ph,colWidths=[5*cm,W-5*cm]); ptbl.setStyle(ts()); story.append(ptbl)
        else: story.append(Paragraph("No tunable hyperparameters.",SB))
        story.append(Spacer(1,.2*cm))
    story.append(PageBreak()); story += [Paragraph("Visual Analysis",SH1),hr()]
    mk = "r2" if task=="Regression" else "accuracy"; ml = "R2" if task=="Regression" else "Accuracy"
    fb = go.Figure([go.Bar(x=[r["name"] for r in results],y=[r.get(mk,0) for r in results],
                            marker=dict(color=[_c(i) for i in range(len(results))],line=dict(width=0)),
                            text=[f"{r.get(mk,0):.4f}" for r in results],textposition="outside")])
    fb.update_layout(title=f"{ml} by Model",height=380)
    story += [Paragraph(f"{ml} Comparison",SH2),_img(fb,W,380),Spacer(1,.3*cm)]
    fr = go.Figure([go.Bar(x=[r["name"] for r in results],y=[r["runtime"] for r in results],
                            marker=dict(color=[_c(i) for i in range(len(results))],line=dict(width=0)),
                            text=[f"{r['runtime']:.3f}s" for r in results],textposition="outside")])
    fr.update_layout(title="Training Runtime (s)",height=320)
    story += [Paragraph("Runtime",SH2),_img(fr,W,320),PageBreak()]
    fi_res = [r for r in results if r.get("feature_importances")]
    if fi_res:
        story += [Paragraph("Feature Importance",SH1),hr()]
        for r in fi_res[:4]:
            df_fi = pd.DataFrame({"Feature":feats,"Importance":r["feature_importances"]}).nlargest(15,"Importance").sort_values("Importance")
            fig_fi = px.bar(df_fi,x="Importance",y="Feature",orientation="h",
                            color_discrete_sequence=[_c(results.index(r))],title=r["name"])
            h_fi = max(260,len(df_fi)*22); fig_fi.update_layout(height=h_fi)
            story += [Paragraph(r["name"],SH2),_img(fig_fi,W,h_fi),Spacer(1,.2*cm)]
    ai = session.get("automl_info")
    if ai:
        story += [PageBreak(),Paragraph("AutoML Focus Results",SH1),hr()]
        fa = ai.get("fit_assessment",{})
        story.append(Paragraph(f"Model: {ai.get('model_name','')} | Best CV {ai.get('metric','')}: {ai.get('best_cv_score',0):.4f} | {ai.get('n_trials',0)} trials",SB))
        if fa.get("verdict"):
            story.append(Paragraph(f"Fit verdict: {fa['verdict'].replace('_',' ').title()} — {fa['recommendation']}",SB))
    story += [Spacer(1,.8*cm),hr(),Paragraph(f"CropML Research Studio v5.0 | {datetime.now().strftime('%Y-%m-%d %H:%M')}",SFT)]
    doc.build(story); return buf.getvalue()
