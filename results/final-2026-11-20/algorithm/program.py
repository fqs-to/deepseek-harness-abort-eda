def compute_tp_gnn_loss(
    positive_scores,
    negative_scores,
    embeddings,
    positive_sources,
    positive_targets,
    negative_sources,
    negative_targets,
    source_batch_size,
    epoch,
    max_epochs,
    graph_scores,
    graph_sources,
    graph_targets,
    graph_net_ids,
    graph_weights,
    net_weights,
    node_features,
    node_areas,
):
    """Augment the maintained random-walk objective with a timing-weighted
    star-edge attraction term. Connected endpoints of heavier timing nets are
    pulled together through the supplied graph dot products; edge importance is
    tanh-bounded and normalized by sqrt(edge count), and the term is ramped in
    after early epochs to preserve initial contrastive geometry. Expected
    measurable effect: lower timing-weighted cut and improved final timing,
    especially TNS, versus this loss's parent. Counted against if matched
    assignments and score show no timing gain, or if WNS/TNS regress.
    """
    contrastive_loss = -(
        F.logsigmoid(positive_scores).sum()
        + F.logsigmoid(-negative_scores).sum()
    ) / max(1, source_batch_size)

    graph_weight_mean = graph_weights.mean()
    graph_weight_scale = torch.clamp(graph_weights.std(), min=1e-6)
    edge_importance = 1.0 + torch.tanh(
        (graph_weights - graph_weight_mean) / graph_weight_scale
    )
    graph_penalty = -(
        edge_importance * F.logsigmoid(graph_scores)
    ).sum() / (float(max(1, len(graph_scores))) ** 0.5)

    warmup = min(
        1.0,
        (float(epoch) + 1.0) / max(1.0, 0.25 * float(max_epochs)),
    )
    return contrastive_loss + warmup * graph_penalty


def decode_tp_gnn_embeddings(embeddings, node_areas, seed, max_iterations, balance_target):
    """Diversify from the parent's frozen reference decoder: replace the seeded
    area-balanced k-means handoff with a deterministic area-weighted spherical
    two-means. The initialization splits on the physical-area-weighted embedding
    axis of largest variance; each bounded iteration recomputes physical-area-
    weighted centers and reassigns whole groups by a cumulative-area balance
    prefix on cosine preference. Expected measurable effect: fewer protected
    closure moves and lower timing-weighted cut, especially on bp-ng45 where the
    parent required 3.5-6% repair movement. Counted against if matched official
    score, WNS/TNS, or repair-moved area does not improve, or if the assignment
    changes globally without timing benefit.
    """
    emb = embeddings
    areas = node_areas
    n = emb.shape[0]
    total_area = areas.sum()
    target_top_area = balance_target * total_area

    mean_vec = (emb * areas[:, None]).sum(axis=0) / total_area
    centered = emb - mean_vec
    var_vec = (centered * centered * areas[:, None]).sum(axis=0) / total_area
    axis_idx = np.argmax(var_vec)
    projection = centered[:, axis_idx]

    proj_order = np.argsort(projection)
    proj_cum = np.cumsum(areas[proj_order])
    proj_k = int(np.argmax(proj_cum >= target_top_area)) + 1
    proj_threshold = projection[proj_order[proj_k - 1]]
    labels_proj = np.where(projection >= proj_threshold, 1, 0)

    labels_index = np.where(np.cumsum(areas) - areas < target_top_area, 1, 0)
    labels_current = np.where(
        (labels_proj.sum() > 0) & (labels_proj.sum() < n),
        labels_proj,
        labels_index,
    )

    for iter_idx in range(max_iterations):
        top_mask = labels_current > 0
        bottom_mask = labels_current < 1
        top_area = (areas * top_mask).sum()
        bottom_area = (areas * bottom_mask).sum()
        safe_top = np.maximum(top_area, 1e-12)
        safe_bottom = np.maximum(bottom_area, 1e-12)

        top_center = (emb * (areas * top_mask)[:, None]).sum(axis=0) / safe_top
        bottom_center = (emb * (areas * bottom_mask)[:, None]).sum(axis=0) / safe_bottom

        top_norm = np.sqrt((top_center * top_center).sum())
        bottom_norm = np.sqrt((bottom_center * bottom_center).sum())
        top_sim = (emb * top_center).sum(axis=1) / np.maximum(top_norm, 1e-12)
        bottom_sim = (emb * bottom_center).sum(axis=1) / np.maximum(bottom_norm, 1e-12)
        preference = top_sim - bottom_sim

        pref_order = np.argsort(-preference)
        pref_cum = np.cumsum(areas[pref_order])
        pref_k = int(np.argmax(pref_cum >= target_top_area)) + 1
        pref_threshold = preference[pref_order[pref_k - 1]]
        labels_next = np.where(preference >= pref_threshold, 1, 0)

        labels_current = np.where(
            (labels_next.sum() > 0) & (labels_next.sum() < n),
            labels_next,
            labels_current,
        )

    return labels_current
