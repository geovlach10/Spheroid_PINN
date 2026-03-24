# 🧬 Antibody Spheroid PINN 

This repository contains a **Physics-Informed Neural Network** (PINN) designed to model the spatiotemporal penetration of antibodies into tumor spheroids.

## 📐 Mathematical Model
The system solves the following PDEs in spherical coordinates:

## pde1

$$
r e s_{1}=\boldsymbol{A} \cdot \varphi \frac{\partial}{\partial \hat{t}}\left(\frac{\hat{C}_{f}}{\varphi}\right)-\boldsymbol{B} \cdot \frac{\partial}{\partial \hat{r}}\left(\hat{r}^{2} \cdot \varphi \cdot \frac{\partial}{\partial \hat{r}}\left(\frac{\hat{C}_{f}}{\varphi}\right)\right)+\boldsymbol{C} \cdot \frac{\hat{C}_{f}}{\varphi} \cdot \hat{r}^{2}-\boldsymbol{D} \cdot \hat{C}_{b} \cdot \hat{r}^{2}
$$

where:

$$
\begin{aligned}
& \hat{C}_{f}=\frac{\left[A B^{(\mathrm{I})}\right]}{\left[A B^{(\mathrm{sol})}\right]} \\
& \hat{t}=\frac{t}{\tau} \\
& \hat{r}=\frac{r}{R} \\
& \boldsymbol{A}=\frac{R^{2} \cdot C_{\mathrm{sol}}}{3600 \cdot \tau}[=] \frac{\mu m^{2} \cdot n M}{\mathrm{sec}} \\
& \boldsymbol{B}=D \cdot C_{\mathrm{sol}}[=] \frac{\mu m^{2} \cdot n M}{\mathrm{sec}} \\
& \boldsymbol{C}=R^{2} \cdot C_{\mathrm{sol}} \cdot R_{t} \cdot \frac{K_{\mathrm{off}}}{K_{D}}[=] \mu m^{2} \cdot n M \cdot n \mathrm{M} \cdot \frac{1}{\mathrm{sec}} \cdot \frac{1}{n \mathrm{M}} \\
& \boldsymbol{D}=R^{2} \cdot C_{\mathrm{sol}} \cdot K_{\mathrm{off}}\left(1+\frac{C_{\mathrm{sol}}}{K_{D}}\right)[=] \mu m^{2} \cdot n M \cdot \frac{1}{\mathrm{sec}} \cdot \frac{n M}{n M}
\end{aligned}
$$

## pde2

$$
r e s_{2}=\boldsymbol{E} \cdot \frac{\partial \hat{C}_{b}}{\partial \hat{t}}-\boldsymbol{F} \cdot \frac{\hat{C}_{t}}{\varphi}+\boldsymbol{G} \cdot \frac{\hat{C}_{t}}{\varphi} \hat{C}_{b}-\boldsymbol{H} \cdot \hat{C}_{b}
$$

where:

$$
\begin{aligned}
\hat{C}_{b} & =\frac{\left[A B^{(b)}\right]}{\left[A B^{(s o l)}\right]} \\
\boldsymbol{E} & =\frac{1}{3600 \cdot \tau}[=] \frac{1}{\sec } \\
\boldsymbol{F} & =\frac{K_{o f f}}{K_{D}} \cdot R_{t}[=] \frac{1}{\sec } \\
\boldsymbol{G} & =\frac{K_{0 f f}}{K_{D}} \cdot C_{\text {sol }}[=] \frac{1}{\sec } \\
\boldsymbol{H} & =K_{\text {off }}-K_{\text {int }}[=] \frac{1}{\sec }
\end{aligned}
$$

## pde3

$$
r e s_{3}=\boldsymbol{Q} \cdot \frac{\partial \hat{C}_{i}}{\partial \hat{t}}-\boldsymbol{T} \cdot \hat{C}_{b}
$$

where:

$$
\begin{aligned}
\hat{c}_{i} & =\frac{\left[A B^{(\mathrm{int})}\right]}{\left[A B^{(\mathrm{sol})}\right]} \\
\boldsymbol{Q} & =\frac{1}{3600 \cdot \tau}[=] \frac{1}{\mathrm{sec}} \\
\boldsymbol{T} & =k_{\mathrm{int}}[=] \frac{1}{\mathrm{sec}}
\end{aligned}
$$

## Boundary Conditions

$$
\begin{aligned}
& \operatorname{res}(r=0)=\left.\frac{\partial}{\partial \hat{r}} \frac{\hat{C}_{t}}{\varphi}\right|_{r=0}-0 \\
& \operatorname{res}(r=R)=\boldsymbol{W} \cdot \varphi \cdot \frac{\partial}{\partial \hat{r}}\left(\frac{\hat{C}_{f}}{\varphi}\right)-P_{\mathrm{ap} / \mathrm{cl}}\left(1-\left.\frac{\hat{C}_{f}}{\varphi}\right|_{r=R}\right)
\end{aligned}
$$

where:
$\boldsymbol{W}=\frac{D}{R}[=] \frac{\mu \mathrm{m}^{2}}{\sec } \frac{1}{\mu \mathrm{~m}} \Leftrightarrow \frac{\mu \mathrm{~m}}{\sec }$

## Initial Conditions

$$
\begin{aligned}
& \operatorname{res}_{f}(t=0)=\hat{c}_{t}-0 \\
& \operatorname{res}_{b}(t=0)=\hat{c}_{b}-0 \\
& \operatorname{res}_{i}(t=0)=\hat{c}_{i}-0
\end{aligned}
$$

## Constants

$$
\begin{aligned}
& R=200 \mu \mathrm{~m} \\
& \tau=50 \mathrm{~h} \\
& {\left[A B^{(\text {sol })}\right]=60 \mathrm{nM}(\text { uptake }), 0 \mathrm{nM} \quad \text { (clearance) }} \\
& D=8.38 \frac{\mu \mathrm{~m}^{2}}{\mathrm{sec}} \\
& K_{\text {off }}=4 \cdot 10^{-3} \cdot \mathrm{sec}^{-1} \\
& K_{D}=6.76 \cdot \mathrm{nM} \\
& K_{\text {int }}=1.4 \cdot 10^{-5} \mathrm{sec}^{-1} \\
& R_{t}=1060 \mathrm{nM} \\
& P_{u p}=2.5 \cdot 10^{-4} \frac{\mu \mathrm{~m}}{\mathrm{sec}} \\
& P_{c 1}=2.6 \cdot 10^{-1} \frac{\mu \mathrm{~m}}{\mathrm{sec}}
\end{aligned}
$$



## 🚀 Getting Started
1. **Environment**: `source venv/bin/activate`
2. **Install**: `pip install -r requirements.txt`


## 📊 Monitoring
* **Reports**: Multi-sheet Excel diagnostics in `results/`.
