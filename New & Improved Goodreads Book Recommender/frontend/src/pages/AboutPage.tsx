import type React from "react";
import codeLinks from "../codeLinks.json";
import { TeX } from "../components/TeX";

/** About: what this is, the data, how recommendations are made, how well it works, privacy, credits. */
export const REPO_URL =
  "https://github.com/mraottth/projects/tree/main/New%20%26%20Improved%20Goodreads%20Book%20Recommender";

const CODE_BASE =
  "https://github.com/mraottth/projects/blob/main/New%20%26%20Improved%20Goodreads%20Book%20Recommender/";
type CodeKey = keyof typeof codeLinks;

/** Link to the function (or file) in the repo; anchors come from scripts/code_links.py so they don't drift. */
function Code({ k, children }: { k: CodeKey; children: React.ReactNode }) {
  const target = codeLinks[k];
  const path = target.split("#")[0];
  const href = CODE_BASE.replace("/blob/", path.includes(".") ? "/blob/" : "/tree/") + target;
  return <a className="code-link" href={href} target="_blank" rel="noreferrer" title={target}><code>{children}</code></a>;
}

/** "Code:" footer listing the relevant functions for a technical-details section. */
function CodeRefs({ items }: { items: [CodeKey, string][] }) {
  return (
    <p className="code-refs">
      <span>Code:</span>
      {items.map(([k, label]) => <Code key={k} k={k}>{label}</Code>)}
    </p>
  );
}

/** Collapsed-by-default section for readers who want the math. */
function Tech({ children }: { children: React.ReactNode }) {
  return (
    <details className="tech">
      <summary>Show technical details</summary>
      <div className="tech-body">{children}</div>
    </details>
  );
}

export function AboutPage({ go }: { go: (v: "rate" | "import" | "recs") => void }) {
  return (
    <article className="about">
      <h1>About Shelf Life</h1>
      <p className="lead-left">
        Shelf Life recommends books from your ratings, using millions of ratings from Goodreads readers.
        Upload your Goodreads library or rate a handful of books you know, and it finds books that readers with your
        taste loved, with a reason for each pick and an estimate of how many stars you&apos;d give it.
      </p>
      <p>
        <a className="primary-link" href={REPO_URL} target="_blank" rel="noreferrer">View the code on GitHub ↗</a>
        {" "}·{" "}
        <button type="button" className="link" onClick={() => go("rate")}>Rate some books</button>
        {" "}·{" "}
        <button type="button" className="link" onClick={() => go("import")}>Import from Goodreads</button>
      </p>

      <h2>The data</h2>
      <p>
        Everything comes from the <a href="https://mengtingwan.github.io/data/goodreads" target="_blank" rel="noreferrer">UCSD
        Book Graph</a>, a research dataset collected by Mengting Wan and Julian McAuley at UC San Diego from public
        Goodreads shelves in late 2017.
      </p>
      <ul>
        <li><strong>15.7 million ratings and reviews</strong> from 465,000 readers.</li>
        <li><strong>2.36 million book editions.</strong> Editions of the same book (hardcover, paperback, translations)
          are merged into one &ldquo;work,&rdquo; so ratings for any edition count toward the same book.</li>
        <li><strong>A catalog of about 105,000 books</strong>: works with at least 20 readers in the data and an English
          (or unlabeled) edition. Covers, descriptions and Goodreads averages come from the same scrape.</li>
        <li><strong>Genres come from readers&apos; own shelves.</strong> Goodreads doesn&apos;t publish genres, so the tags
          readers file books under (&ldquo;cozy-mystery,&rdquo; &ldquo;space-opera,&rdquo; &ldquo;ww2&rdquo;) were
          mapped to 205 descriptive tags and 34 parent genres.</li>
      </ul>
      <Tech>
        <ul>
          <li><strong>Source files:</strong> <code>goodreads_books</code>, <code>book_works</code>, <code>book_authors</code>,
            <code>book_genres_initial</code> and <code>reviews_dedup</code> (15.74M rows), streamed from gzipped JSON lines.</li>
          <li><strong>Editions → works:</strong> each edition&apos;s <code>book_id</code> maps to its <code>work_id</code>. When a reader
            rated several editions of one work, the max rating is kept. A rating of 0 means &ldquo;read, not rated&rdquo; and is
            kept as a weak implicit signal.</li>
          <li><strong>Catalog:</strong> works with ≥ 20 distinct 1–5★ raters (105,230 works), at least one English or unlabeled
            edition, and a mostly Latin-script title. The display edition is Goodreads&apos; <code>best_book_id</code>. Series name
            and position are parsed from titles like &ldquo;Title (Series, #3)&rdquo;, and ranges and split editions are flagged as box sets.</li>
          <li><strong>Training matrix:</strong> users with ≥ 3 catalog ratings, minus 10,000 held-out test users: 277,141 users ×
            105,230 books, 10.39M ratings plus 0.29M read-unrated.</li>
          <li><strong>Genres:</strong> shelf counts are summed across editions. The 500 most common non-status shelves were mapped
            to display names and 34 parents in a reviewed YAML file. Tag weight = shelf share × log(N / document frequency), so
            specific tags beat ubiquitous ones. A book&apos;s parent genre is the parent with the largest summed shelf share,
            falling back to UCSD&apos;s coarse genre votes (100% coverage; 99.7% of books have tags).</li>
        </ul>
        <CodeRefs items={[["books_stream", "s01_books.main()"], ["ratings_to_works", "s03_ratings.main()"], ["catalog", "s04_catalog.main()"], ["parse_series", "parse_series()"], ["matrix_split", "s05_matrix.main()"], ["genres_apply", "s06_genres.apply()"], ["genre_map", "shelf_genres.yaml"], ["pipeline_config", "pipeline.yaml"]]} />
      </Tech>

      <p className="note">
        <strong>Limits:</strong> nothing published after 2017 is included, so newer books in an upload can&apos;t be matched.
        The data also reflects the people who rate and review on Goodreads, who read more young adult and romance than
        the population at large. Young adult books (about 11% of the catalog, flagged by genre and readers&apos; shelves)
        are left out of recommendations unless you tick <em>Include young adult books</em> in the filters.
      </p>

      <h2>How recommendations are made</h2>
      <p>Two models score every book, and their scores are blended based on how many books you&apos;ve rated.</p>
      <dl className="method">
        <div>
          <dt>Similar books</dt>
          <dd>
            For every book, the 50 books whose readers rated it most similarly were precomputed from 10 million ratings.
            Your highly rated books &ldquo;vote&rdquo; for their neighbors and your low ratings vote against theirs. This
            works well even from a handful of ratings, and it&apos;s where each &ldquo;Because you liked…&rdquo; comes from.
            <Tech>
              <p>Adjusted cosine similarity on user-mean-centered ratings, shrunk by co-rating support:</p>
              <TeX block>{String.raw`\mathrm{sim}(i,j)=\cos(\mathbf{r}'_i,\mathbf{r}'_j)\cdot\frac{n_{ij}}{n_{ij}+25},\qquad r'_{ui}=r_{ui}-\bar r_u,\quad n_{ij}\ge 5`}</TeX>
              <p>
                The top 50 neighbors per book are kept (int32 indices plus float16 similarities, about 32 MB). Computing them
                takes 3.3 minutes on a laptop: sparse <TeX>{String.raw`X^{\top}X`}</TeX> products in blocks of 500 books, with a 1.7 GB peak.
                At request time each rated book votes with weight <TeX>{String.raw`w_i=r_i-\tilde b_u`}</TeX>, where{" "}
                <TeX>{String.raw`\tilde b_u=\frac{\sum r\,+\,3\cdot 5}{n+5}`}</TeX> is the user&apos;s mean shrunk toward 3★. So a single 5★ rating
                still counts as a strong positive, and a 1–2★ rating pushes its neighbors down.
              </p>
              <TeX block>{String.raw`\mathrm{score}_{\mathrm{ii}}(j)=\sum_{i\,\in\,\mathrm{rated}} w_i\,\mathrm{sim}(i,j)`}</TeX>
              <p>&ldquo;Because you liked&rdquo; lists the liked books in <em>j</em>&apos;s own neighbor list with the largest{" "}
                <TeX>{String.raw`w_i\cdot\mathrm{sim}`}</TeX>, falling back to the nearest liked book in ALS space.</p>
              <CodeRefs items={[["item_knn", "item_knn()"], ["ii_weights", "item_item_weights()"], ["ii_scores", "item_item_scores()"], ["explain", "explain()"]]} />
            </Tech>
          </dd>
        </div>
        <div>
          <dt>Taste model</dt>
          <dd>
            A matrix-factorization model (implicit ALS) places every book and reader as a point in a 64-dimensional
            &ldquo;taste space.&rdquo; Your point is calculated on the fly from your ratings, and books near it score
            highly. It picks up broader patterns than similar books do, and gets better the more you rate.
            <Tech>
              <p>
                Implicit-feedback ALS (Hu, Koren &amp; Volinsky 2008) via the <code>implicit</code> library: 64 factors,
                λ = 1.0, 15 iterations, trained on 9.64M nonzeros in 36 s. Ratings become preferences with graded confidence:
              </p>
              <TeX block>{String.raw`\begin{gathered}p_{ui}=\begin{cases}1 & \text{rated} \ge 3\text{★ or read}\\ 0 & \text{otherwise}\end{cases}\qquad c_{ui}=1+\alpha\,g(r_{ui}),\quad \alpha=30\\[4pt] g(5\text{★})=1.0,\quad g(4\text{★})=0.7,\quad g(3\text{★})=0.3,\quad g(\text{read, unrated})=0.2\quad(1\text{–}2\text{★ are not positives})\end{gathered}`}</TeX>
              <p>A new user is <strong>folded in</strong> without retraining: with item factors <em>Y</em> fixed, the user vector
                is the exact least-squares solution (the same as <code>implicit</code>&apos;s <code>recalculate_user</code>). That&apos;s a
                64×64 solve, under a millisecond:</p>
              <TeX block>{String.raw`\mathbf{u}=\left(Y^{\top}Y+Y^{\top}(C_u-I)\,Y+\lambda I\right)^{-1}Y^{\top}C_u\,\mathbf{p}_u,\qquad \mathrm{score}_{\mathrm{als}}(j)=\mathbf{y}_j^{\top}\mathbf{u}`}</TeX>
              <p><TeX>{String.raw`Y^{\top}Y`}</TeX> is precomputed. Hyperparameters were picked from a sweep over factors × λ × α. The blended
                NDCG barely moved across settings (0.1212–0.1220), while ALS-only NDCG favored α = 30, λ = 1.0, which also
                sharpens the user vectors behind &ldquo;readers like you.&rdquo;</p>
              <CodeRefs items={[["als_confidence", "confidence_matrix()"], ["als_train", "train_als()"], ["fold_in", "fold_in()"], ["als_sweep", "tune_als.py"]]} />
            </Tech>
          </dd>
        </div>
        <div>
          <dt>The blend</dt>
          <dd>
            With a few ratings the ranking leans on similar books; as you rate more, it shifts toward the taste model.
            A small popularity adjustment helps new users and is dialed down for people with long histories, so heavy
            readers don&apos;t just get bestsellers. Well-known books we predict you&apos;d rate highly get a boost, so a
            widely loved classic you&apos;d probably enjoy isn&apos;t buried; obscure books don&apos;t get it, because their high
            predictions mostly just reflect a high average. Books we predict you&apos;d rate noticeably below your own average are
            left out of For you, and later volumes of a series are hidden unless you&apos;ve read the one before.
            <Tech>
              <p>Candidates are the union of each model&apos;s top 300 after filters. Each signal is z-scored within that
                candidate set and combined with weights that depend on the number of ratings <em>n</em>:</p>
              <TeX block>{String.raw`\begin{gathered}\mathrm{final}=a\,z(\mathrm{als})+(1-a)\,z(\mathrm{ii})+\beta\,z\big(\log(1+\mathrm{readers})\big)+a\,\delta\,f\,z(\hat r)\\[4pt] a(n)=\frac{n}{n+20},\qquad \beta(n)=(1-a)\cdot 0+a\cdot(-0.3),\qquad f=\mathrm{clip}\Big(\frac{\log_{10}\mathrm{ratings}-4}{1.5},0,1\Big)\end{gathered}`}</TeX>
              <p>
                k<sub>a</sub> = 20 came from a grid search. Popularity helped cold-start users slightly and hurt users with full
                histories; a penalty of β = −0.3 raised full-history NDCG@20 from 0.128 to 0.132 and increased catalog
                coverage, which is why β is interpolated rather than fixed. A Bayesian quality prior (γ) didn&apos;t help and is
                off. The last term is the prediction boost: <TeX>{String.raw`\hat r`}</TeX> is your predicted rating, δ = 0.75,
                and the fame weight <em>f</em> goes from 0 at 10k Goodreads ratings to 1 at about 316k. The 50 best
                fame-weighted predictions are also added to the candidate pool. On held-out users it costs about 4% of
                full-history NDCG@20 (about 2% with 10 ratings); without the fame gate the same boost cost about 14%.
                &ldquo;Sort by predicted rating&rdquo; re-orders the same candidate pool. Series rule: hide{" "}
                <code>series_pos &gt; 1</code> unless it&apos;s the lowest unread volume after the furthest one you&apos;ve read.
              </p>
              <p>
                <strong>Prediction floor.</strong> Once you have 5+ ratings, For you excludes books whose calibrated predicted
                rating is below your average rating minus 0.25★, before the candidate pool is built (so the pool refills
                with books above the floor). Ranking on the blend alone sometimes put a book the rating model expects you to
                dislike near the top, because the blend also rewards what readers like you read. The floor costs accuracy
                on the held-out metric, which rewards finding books you&apos;d read <em>and</em> love (NDCG@20, 1,200 users):
              </p>
              <table className="viz-table about-table">
                <thead><tr><th>Visible ratings</th><th>5</th><th>10</th><th>25</th><th>all</th></tr></thead>
                <tbody>
                  <tr><td>No floor</td><td>0.082</td><td>0.106</td><td>0.126</td><td>0.144</td></tr>
                  <tr><td><strong>Floor at average − 0.25 (used)</strong></td><td>0.076</td><td>0.102</td><td>0.120</td><td>0.139</td></tr>
                  <tr><td>Floor at average</td><td>0.063</td><td>0.093</td><td>0.113</td><td>0.131</td></tr>
                </tbody>
              </table>
              <p>The −0.25 margin keeps most of the benefit for about a 4% cost, versus about 10% for a floor exactly at the
                average. The Popular, Top rated and To-read lists aren&apos;t floored.</p>
              <CodeRefs items={[["blend", "blend()"], ["fame_weight", "fame_weight()"], ["prediction_floor", "prediction_floor()"], ["filter_mask", "filter_mask()"], ["next_in_series", "next_in_series()"], ["recommend", "recommend()"]]} />
            </Tech>
          </dd>
        </div>
        <div>
          <dt>Readers like you</dt>
          <dd>
            The 300 readers closest to you in taste space. &ldquo;Popular&rdquo; is what they read most (adjusted so
            universal bestsellers don&apos;t crowd everything out), &ldquo;Top rated&rdquo; is what they rated highest, and
            every book shows their average rating. This unlocks after 5 ratings.
            <Tech>
              <p>
                Neighbors are found by cosine similarity between your folded-in ALS vector and the L2-normalized vectors of the
                149,734 training readers with ≥ 10 ratings (a single 150k × 64 mat-vec, a few ms). No per-request
                nearest-neighbor search over the ratings matrix is needed, which is what made the 2023 version take a minute.
                The top M = 300 by cosine <em>s<sub>v</sub></em> are kept, and their actual shelves are read from a users × books
                int8 matrix (27 MB).
              </p>
              <TeX block>{String.raw`\begin{gathered}\mathrm{popular}(j)=\frac{\mathrm{reach}(j)}{\mathrm{globalRate}(j)^{0.5}},\qquad \mathrm{reach}(j)=\frac{\sum_v s_v\,\mathbb{1}[v\text{ read }j]}{\sum_v s_v}\\[6pt] \mathrm{topRated}(j)=\frac{\sum_v s_v\,r_{vj}+5\,\mu_j}{\sum_v s_v+5},\quad\text{shown if}\ \ge\max(5,\,M/100)\ \text{neighbors rated } j\end{gathered}`}</TeX>
              <p>The &ldquo;% of similar readers read it&rdquo; and the &ldquo;Readers like you&rdquo; average on each card are
                unweighted over the 300 neighbors, so they&apos;re easy to interpret. The damping exponent keeps universally read
                books from filling the popular list.</p>
              <CodeRefs items={[["neighbors", "neighbors()"], ["similar_readers", "similar_readers()"]]} />
            </Tech>
          </dd>
        </div>
        <div>
          <dt>Predicted rating</dt>
          <dd>
            A separate estimate of the stars you&apos;d give a book: its average, adjusted for how tough or generous a
            rater you are, plus a correction from similar books you&apos;ve rated. It&apos;s then spread out toward how
            you actually rate, in proportion to how much of the prediction comes from your own ratings: books closely
            tied to ones you&apos;ve rated can reach the top of your scale, while a book with no connection to your
            reading stays near its average adjusted for you, rather than having small differences between book
            averages blown up.
            <Tech>
              <p>A baseline plus a neighborhood residual (a classic item-kNN predictor), separate from the ranking blend:</p>
              <TeX block>{String.raw`\begin{aligned}\mu_j&=\dfrac{n_j\,\bar r_j+50\,g_j}{n_j+50}&&\text{dataset mean, shrunk toward the Goodreads average }g_j\\[4pt] b_u&=\dfrac{\sum_i\,(r_{ui}-\mu_i)}{n+5}&&\text{user bias, shrunk}\\[4pt] \hat r_{uj}&=\operatorname{clip}_{[1,5]}\!\left(\mu_j+b_u+\dfrac{\sum_i s_{ij}\,(r_{ui}-\mu_i-b_u)}{\sum_i|s_{ij}|+0.5}\right)\end{aligned}`}</TeX>
              <p>The sum runs over your rated books linked to <em>j</em> in either direction of the top-50 neighbor lists, using
                the larger similarity. Accuracy of this raw prediction on held-out ratings (RMSE in stars, lower is better)
                improves as you rate more:</p>
              <table className="viz-table about-table">
                <thead><tr><th>Visible ratings</th><th>1</th><th>3</th><th>5</th><th>10</th><th>25</th><th>all</th></tr></thead>
                <tbody>
                  <tr><td>Predicted rating</td><td>0.935</td><td>0.910</td><td>0.900</td><td>0.883</td><td>0.869</td><td><strong>0.852</strong></td></tr>
                  <tr><td>Book mean <TeX>{String.raw`\mu_j`}</TeX></td><td colSpan={6}>0.953 (doesn&apos;t use your ratings)</td></tr>
                  <tr><td>Goodreads average alone</td><td colSpan={6}>0.971</td></tr>
                </tbody>
              </table>
              <p>
                <strong>Calibration.</strong> A least-squares predictor regresses toward your mean, so a tough grader almost
                never sees a 4.5 (0.2% of held-out predictions for raters averaging under 3.6★, who actually give 5★ 14% of
                the time). The displayed value is therefore quantile-matched to your own ratings. Take raw predictions for the
                books you rated as the reference distribution <TeX>{String.raw`F_{\mathrm{ref}}`}</TeX>, and your star histogram,
                shrunk toward the dataset&apos;s with the weight of 10 ratings, as the target, with each star spread uniformly
                over a band (5★ = [4.5, 5], 4★ = [3.5, 4.5], …). Each book moves toward that map only in proportion to its
                evidence <TeX>{String.raw`e_j`}</TeX>, the share of its prediction that rests on similar books you rated:
              </p>
              <TeX block>{String.raw`\tilde r_{uj}=\hat r_{uj}+e_j\,w\Big(Q_{\mathrm{you}}\!\big(F_{\mathrm{ref}}(\hat r_{uj})\big)-\hat r_{uj}\Big),\qquad w=\frac{n}{n+10},\qquad e_j=\frac{\sum_i|s_{ij}|}{\sum_i|s_{ij}|+0.1}`}</TeX>
              <p>
                Beyond the reference range the map approaches 1 and 5 exponentially, so nothing is pinned at 5. Because the
                stretch varies by book, sort by predicted rating uses the displayed value. Stretching everything fully spread
                predictions like your ratings but amplified noise: two books with no link to your reading could land a full
                star apart just because their averages differed by 0.25, and RMSE rose from 0.86 to 0.99. With the
                evidence weighting, on 1,200 held-out users, the displayed RMSE is 0.87, and 22% of predictions reach 4.5
                or more (vs. 35% actual 5★), mostly books tied to your own ratings. Below 5 ratings the raw value is shown.
              </p>
              <CodeRefs items={[["predict_ratings", "predict_ratings()"], ["calibration", "calibration()"], ["rating_metrics", "rating_metrics()"]]} />
            </Tech>
          </dd>
        </div>
      </dl>

      <h2>How well it works</h2>
      <p>
        About 10,000 readers were set aside and never used to train anything. For each of them, 30% of their ratings
        were hidden, and the question was whether the books they loved (4–5★) showed up in their top 20. Scores are
        NDCG@20, where higher is better:
      </p>
      <table className="viz-table about-table">
        <thead><tr><th>Method</th><th>1 rating</th><th>3 ratings</th><th>10 ratings</th><th>Full history</th></tr></thead>
        <tbody>
          <tr><td><strong>Shelf Life (blend)</strong></td><td>0.042</td><td><strong>0.064</strong></td><td><strong>0.102</strong></td><td><strong>0.139</strong></td></tr>
          <tr><td>Similar books only</td><td>0.038</td><td>0.061</td><td>0.095</td><td>0.134</td></tr>
          <tr><td>Taste model only</td><td>0.042</td><td>0.059</td><td>0.084</td><td>0.110</td></tr>
          <tr><td>Most popular books</td><td>0.032</td><td>0.032</td><td>0.033</td><td>0.036</td></tr>
        </tbody>
      </table>
      <Tech>
        <ul>
          <li><strong>Split:</strong> 10,000 users with ≥ 10 catalog ratings were sampled (seed 42) before any training and
            excluded from the similar-books lists, ALS, and the readers-like-you pool. The eval folds them in exactly as the app
            folds in a new visitor, through the same serving code.</li>
          <li><strong>Protocol:</strong> 30% of each test user&apos;s ratings are hidden at random. Visible ratings are truncated
            to n ∈ {"{1, 3, 5, 10, 25, all}"}, and relevant items are hidden ratings ≥ 4★. There&apos;s binary-relevance
            NDCG@20 over the full catalog minus visible books, with content filters off. 9,906 users have at least one
            relevant hidden item.</li>
          <li><strong>Tuning caveat:</strong> blend weights (k<sub>a</sub>, β, γ) and ALS settings were chosen on 800–2,000-user
            subsamples of this same held-out pool, so the reported numbers are mildly optimistic. A clean version would carve
            out a separate validation split.</li>
          <li><strong>Also reported</strong> in <code>eval/reports/</code>: recall@20, catalog coverage and the mean log-popularity
            of recommendations (a popularity-bias check).</li>
        </ul>
        <CodeRefs items={[["matrix_split", "s05_matrix.main()"], ["split_users", "split_users()"], ["evaluate", "evaluate()"], ["metrics", "metrics()"], ["eval_reports", "eval/reports/"]]} />
      </Tech>

      <p>
        Predicted ratings are typically within about 0.86 stars of the rating people actually gave (RMSE, full
        history), compared with 0.97 for just using the book&apos;s average.
      </p>

      <Tech>
        <p>
          <strong>Serving:</strong> an offline pipeline (11 scripts, about 15 minutes on a laptop) writes about 311 MB of
          artifacts: numpy arrays, a sparse matrix, and a SQLite catalog with FTS5 full-text search. A FastAPI server loads
          them in about 0.4 s and needs only numpy/scipy at request time. p95 latency: search 3 ms, recommendations 13 ms
          (uncached, including readers like you), a 1,000-row CSV import 28 ms, with about 250 MB of RAM. Per-user model
          scores are LRU-cached by a hash of the ratings, so changing filters or tabs only re-filters. The frontend is React +
          TypeScript. Your shelf lives in localStorage, and filters live in the URL.
        </p>
        <CodeRefs items={[["package", "s10_package.main()"], ["load_artifacts", "load_artifacts()"], ["api_recommend", "recommend_route()"], ["api_search", "Catalog.search()"], ["csv_matching", "match_row()"], ["shelf_store", "ShelfProvider"]]} />
      </Tech>

      <h2>The Assistant</h2>
      <p>
        The Assistant is powered by Claude (Anthropic). It doesn&apos;t replace the recommender; it uses it. When
        you ask for a book, it pulls your For you list (with any filters it needs), what readers like you read and loved,
        your to-read list, or details about a specific book, then chooses and explains picks in plain language. It also
        knows books published after 2017, which aren&apos;t in the dataset, and can search the web for them. Those books have
        no cover card or predicted rating, since the recommender has never seen them. It can also just talk about books.
      </p>
      <Tech>
        <p>
          <strong>How it&apos;s wired:</strong> a quick Claude Haiku call sorts each message into simple or complex; simple ones are answered by Haiku 4.5 and complex ones (newer books, taste analysis, book discussion) by Sonnet 5. Either way, the model runs a tool-use loop on the server. The tools call the same
          functions as the rest of the app, so ranks and predicted ratings match what you see elsewhere. The system prompt
          carries a digest of your library (every rated book by star, your to-read list, and books from your Goodreads
          export that aren&apos;t in the catalog) and is prompt-cached, so follow-up messages are cheap. Replies stream to the
          page as server-sent events; catalog books come back as markers the page turns into cover cards. Each visitor has
          a daily message cap.
        </p>
        <CodeRefs items={[["chat_tools", "Toolbox"], ["chat_loop", "run_chat()"], ["chat_digest", "library_digest()"]]} />
      </Tech>

      <h2>Privacy</h2>
      <p>
        There are no accounts. Your ratings are saved only in this browser and sent with each request so the server
        can score books; nothing is stored on the server. An uploaded Goodreads file is read to match your books and
        then discarded. &ldquo;Clear my shelf&rdquo; at the bottom of any page erases everything. If you use the Assistant, your
        ratings, reviews and shelves are sent to Anthropic along with each message to generate the reply.
      </p>

      <h2>Credits</h2>
      <p className="small">
        Data: Mengting Wan and Julian McAuley, &ldquo;Item Recommendation on Monotonic Behavior Chains,&rdquo; RecSys 2018;
        Mengting Wan, Rishabh Misra, Ndapa Nakashole and Julian McAuley, &ldquo;Fine-Grained Spoiler Detection from
        Large-Scale Review Corpora,&rdquo; ACL 2019. The dataset is for non-commercial use. Built with FastAPI, NumPy,
        SciPy, implicit and React.
      </p>
    </article>
  );
}
