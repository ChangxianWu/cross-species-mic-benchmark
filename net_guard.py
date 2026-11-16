# -*- coding: utf-8 -*-
"""Minimal outbound-HTTP guard for the download steps.

A request is only sent to a host on a short allowlist, over https, and only if the
hostname resolves to a public IP. Redirects are re-checked hop by hop, so a public
URL cannot bounce the client into a private or link-local address.

The point is narrow: the download scripts should talk to the documented data
sources and nothing else, including when a server answers with a redirect.
"""
import ipaddress
import socket
import ssl
import urllib.parse
import urllib.request

# Only the documented data sources (plus the FTP path the genotype tables come
# from, which is fetched manually and never by these scripts).
ALLOWED_HOSTS = {
    "www.bv-brc.org",
    "bv-brc.org",
    "ftp.ncbi.nlm.nih.gov",
}


class BlockedURL(ValueError):
    """URL failed the protocol / host / resolved-IP checks; no request is sent."""


def _ssl_context():
    """Prefer certifi's CA bundle: it is updated with the package and is more
    current than the interpreter's default store on Windows."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def assert_safe_url(url):
    """Check scheme, host allowlist and resolved IP; return the url or raise."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https":
        raise BlockedURL("only https is allowed, got %r" % (parts.scheme,))
    host = (parts.hostname or "").lower()
    if host not in ALLOWED_HOSTS:
        raise BlockedURL("host not in allowlist: %r" % (host,))
    try:
        infos = socket.getaddrinfo(host, parts.port or 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise BlockedURL("cannot resolve %s: %s" % (host, exc))
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                or ip.is_multicast or ip.is_unspecified):
            raise BlockedURL("resolved to non-public address for %s: %s" % (host, ip))
    return url


class _AllowlistRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-validate every redirect target instead of trusting the first hop."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        assert_safe_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(
    _AllowlistRedirectHandler,
    urllib.request.HTTPSHandler(context=_ssl_context()),
)


def open_url(url, headers=None, timeout=45, data=None):
    """Validate, then open. Returns a response usable as a context manager.

    `data` goes into the request body only; the URL is still validated, so a POST
    cannot be used to reach a host outside the allowlist.
    """
    assert_safe_url(url)
    return _OPENER.open(
        urllib.request.Request(url, headers=headers or {}, data=data),
        timeout=timeout)
