# Literature and data foundations

## Data

- **PFF FC 2022 FIFA World Cup dataset.** Free, request-based access through PFF FC. Broadcast-derived tracking at about 30 Hz (the ball is effectively closer to 15 Hz), with metadata, rosters and event data for all 64 matches. Roughly 27% of broadcast airtime has no usable tracking (replays, close-ups, ball out of play).
- **DFL/IDSSE open data.** Bassek, M. et al. (2025). *An integrated dataset of spatiotemporal and event data in elite soccer.* Scientific Data, 12. https://doi.org/10.1038/s41597-025-04505-y. Seven German league matches from 2022/23, optical TRACAB tracking at 25 Hz plus DFL event data, CC-BY 4.0 (© DFL). Files: https://doi.org/10.6084/m9.figshare.28196177
- **Metrica Sports open data.** Three anonymised matches of tracking and event data. The Dangerous Accessible Space model was fitted and validated on it.

## Defensive phases and structure

- Sarmento, H. et al. (2014). *Match analysis in football: a systematic review.* Journal of Sports Sciences, 32(20), 1831–1843. The four-moment framework: offensive and defensive organisation, offensive and defensive transition.
- Casal, C. A. et al. (2016). *Identification of defensive performance factors in the 2010 FIFA World Cup South Africa.* Sports, 4(4), 54.
- Forcher, L. et al. (2022). *The use of player tracking data to analyze defensive play in professional soccer: a scoping review.* International Journal of Sports Science & Coaching.
- Forcher, L. et al. (2023). *Prediction of defensive success in elite soccer using machine learning.* Science and Medicine in Football, 8(4).
- Forcher, L. et al. (2024). *The keys of pressing to gain the ball.* Science and Medicine in Football, 8, 161–169.
- Forcher, L. et al. *Is a compact organization important for defensive success in elite soccer?* Science and Medicine in Football. It found that compactness of the five defenders nearest the ball matters more than whole-team compactness, which is the motivation for the TIGHT/LOOSE tag. (Volume and pages not yet verified.)
- FIFA, *Enhanced Football Intelligence* (technical documentation, 2022). An automated frame-level classifier used at the 2022 World Cup (high press, mid block, low block, counterpress, recovery, defensive transition). It's the closest existing precedent to this model.

## Pressure

- Bekkers, J. (2025). *Pressing Intensity: an intuitive measure for pressing in soccer.* arXiv:2501.04712. Time-to-intercept turned into a probability with a logistic function; `pressure.py` implements a single-target version.
- Merckx, S. et al. (2021). *Measuring the effectiveness of pressing in soccer.* MLSA Workshop, ECML-PKDD.
- Andrienko, G. et al. (2017). *Visual analysis of pressure in football.* Data Mining and Knowledge Discovery, 31(6), 1793–1839.
- Bauer, P. & Anzer, G. (2021). *Data-driven detection of counterpressing in professional football.* Data Mining and Knowledge Discovery, 35(5), 2009–2049. It shows there is no single clean duration for a transition, which is why `transitions.py` uses a settling detector instead of a fixed timer.

## Space and value

- Fernández, J. & Bornn, L. (2018). *Wide Open Spaces: a statistical technique for measuring space creation in professional soccer.* MIT Sloan Sports Analytics Conference. Influence-function pitch control plus a value surface learned from proprietary LaLiga tracking. It wasn't reproducible here: the training data was never released, and the paper attributes space to attackers rather than describing what a defence leaves open.
- Bischofberger, J. & Baca, A. (2026). *Dangerous accessible space: a unified model of space and value in team sports.* Journal of Big Data, 13(1), 76. A physics-based pass-completion map times an expected-goals location value, designed to work from small open datasets. It's the basis of the danger heatmap, via the `accessible-space` package.
