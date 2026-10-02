-- Book dimension for analytics, dashboarding and RAG preparation.
-- Grain: one row per book_id.

with books as (
    select
        * exclude (description),
        trim(regexp_replace(
            regexp_replace(coalesce(description, ''), '<[^>]+>', ' ', 'g'),
            '&(amp|lt|gt|nbsp|quot);', ' ', 'g'
        )) as description,
        coalesce(nullif(trim(title), ''), '[Unknown title]') as cleaned_title,
        case
            when title is null or trim(title) = '' then true
            else false
        end as is_title_missing,
        case
            when try_cast(pages as integer) between 1 and 10000
            then try_cast(pages as integer)
            else null
        end as pages_clean,
        case
            when try_cast(release_year as integer)
                between -5000 and year(current_date) + 1
            then try_cast(release_year as integer)
            else null
        end as release_year_clean,
        case
            when pages is not null
            and (
                try_cast(pages as integer) is null
                or try_cast(pages as integer) <= 0
                or try_cast(pages as integer) > 10000
            )
            then true
            else false
        end as is_pages_suspicious,
        case
            when release_year is not null
            and (
                try_cast(release_year as integer) is null
                or try_cast(release_year as integer) < -5000
                or try_cast(release_year as integer) > year(current_date) + 1
            )
            then true
            else false
        end as is_release_year_suspicious
    from {{ source('silver', 'silver_books') }}
),

author_rows as (
    select
        ba.book_id,
        a.id as author_id,
        trim(a.name) as author_name,
        string_agg(
            distinct nullif(trim(ba.role), ''),
            ', ' order by nullif(trim(ba.role), '')
        ) as roles
    from {{ source('silver', 'silver_book_authors') }} ba
    inner join {{ source('silver', 'silver_authors') }} a
        on a.id = ba.author_id
    where ba.book_id is not null
      and ba.author_id is not null
      and a.name is not null
      and trim(a.name) <> ''
    group by ba.book_id, a.id, trim(a.name)
),

authors as (
    select
        book_id,
        string_agg(
            author_name,
            ', ' order by author_name
        ) as author_names,
        string_agg(
            author_name
                || case
                    when roles is not null then ' (' || roles || ')'
                    else ''
                end,
            ', ' order by author_name
        ) as author_credits
    from author_rows
    group by book_id
),

series_rows as (
    select distinct
        bs.book_id,
        s.id as series_id,
        trim(s.name) as series_name,
        bs.position
    from {{ source('silver', 'silver_book_series') }} bs
    inner join {{ source('silver', 'silver_series') }} s
        on s.id = bs.series_id
    where bs.book_id is not null
      and bs.series_id is not null
      and s.name is not null
      and trim(s.name) <> ''
),

series as (
    select
        book_id,
        string_agg(
            series_name
                || case
                    when position is not null
                        then ' #' || cast(position as varchar)
                    else ''
                end,
            ', ' order by position nulls last, series_name
        ) as series_info
    from series_rows
    group by book_id
),

tag_votes as (
    select
        bt.book_id,
        trim(t.tag) as tag_name,
        t.tag_category,
        count(*) as vote_count
    from {{ source('silver', 'silver_book_taggings') }} bt
    inner join {{ source('silver', 'silver_tags') }} t
        on t.id = bt.tag_id
    where bt.book_id is not null
      and bt.tag_id is not null
      and t.tag is not null
      and trim(t.tag) <> ''
      and t.tag_category is not null
    group by bt.book_id, trim(t.tag), t.tag_category
),

ranked_tags as (
    select
        *,
        row_number() over (
            partition by book_id, tag_category
            order by vote_count desc, tag_name
        ) as category_rank
    from tag_votes
),

tags as (
    select
        book_id,
        string_agg(tag_name, ', ' order by vote_count desc, tag_name)
            filter (
                where tag_category = 'Genre'
                  and category_rank <= 20
            ) as genres,
        string_agg(tag_name, ', ' order by vote_count desc, tag_name)
            filter (
                where tag_category = 'Tag'
                  and category_rank <= 20
            ) as general_tags,
        string_agg(tag_name, ', ' order by vote_count desc, tag_name)
            filter (
                where tag_category = 'Mood'
                  and category_rank <= 20
            ) as moods,
        string_agg(tag_name, ', ' order by vote_count desc, tag_name)
            filter (
                where tag_category = 'Content Warning'
                  and category_rank <= 20
            ) as content_warnings,
        string_agg(tag_name, ', ' order by vote_count desc, tag_name)
            filter (
                where tag_category = 'Pace'
                  and category_rank <= 20
            ) as pace
    from ranked_tags
    group by book_id
)

select
    b.id as book_id,
    b.cleaned_title as title,
    b.subtitle,
    b.description,
    b.headline,
    b.literary_type,
    b.book_category,
    b.release_year,
    b.release_year_clean,
    b.is_release_year_suspicious,
    b.pages,
    b.pages_clean,
    b.is_pages_suspicious,
    b.rating,
    b.ratings_count,
    b.ratings_distribution,
    b.slug,

    coalesce(a.author_names, '') as author_names,
    coalesce(a.author_credits, '') as author_credits,   
    coalesce(s.series_info, '') as series_info,
    coalesce(t.genres, '') as genres,
    coalesce(t.general_tags, '') as general_tags,
    coalesce(t.moods, '') as moods,
    coalesce(t.content_warnings, '') as content_warnings,
    coalesce(t.pace, '') as pace,

    b.is_title_missing,
    
    trim(
        concat_ws(
            chr(10),
            'Title: ' || b.cleaned_title,
            case
                when nullif(trim(b.subtitle), '') is not null
                    then 'Subtitle: ' || trim(b.subtitle)
            end,
            case
                when nullif(trim(a.author_names), '') is not null
                    then 'Authors: ' || trim(a.author_names)
            end,
            case
                when nullif(trim(s.series_info), '') is not null
                    then 'Series: ' || trim(s.series_info)
            end,
            case
                when nullif(trim(t.genres), '') is not null
                    then 'Genres: ' || trim(t.genres)
            end,
            case
                when nullif(trim(t.general_tags), '') is not null
                    then 'Tags: ' || trim(t.general_tags)
            end,
            case
                when nullif(trim(t.moods), '') is not null
                    then 'Moods: ' || trim(t.moods)
            end,
            case
                when nullif(trim(t.pace), '') is not null
                    then 'Pace: ' || trim(t.pace)
            end,
            case
                when nullif(trim(t.content_warnings), '') is not null
                    then 'Content warnings: ' || trim(t.content_warnings)
            end,
            case
                when nullif(trim(b.headline), '') is not null
                    then 'Headline: ' || trim(b.headline)
            end,
            case
                when nullif(trim(b.description), '') is not null
                    then 'Description: ' || trim(b.description)
            end
        )
    ) as chunk_text,
    b.compilation,
    b.is_partial_book,
    case
    when length(b.description) >= 50
        and (
            nullif(trim(t.genres), '')          is not null
            or nullif(trim(a.author_names), '') is not null
            or nullif(trim(t.moods), '')        is not null
            or nullif(trim(s.series_info), '')  is not null
        )
            then 'Rich'
        when length(b.description) >= 50
            then 'Metadata'
        when nullif(trim(a.author_names), '') is not null
        or nullif(trim(t.genres), '')       is not null
            then 'Metadata'
        else 'Poor'
    end as content_quality,
    b.updated_at
from books b
left join authors a
    on a.book_id = b.id
left join series s
    on s.book_id = b.id
left join tags t
    on t.book_id = b.id
